"""Unit tests for virtual hand UDP port 5005 communication.

Validates:
1. Python socket.socket(AF_INET, SOCK_DGRAM) send/receive on port 5005.
2. JSON serialization and deserialization for binary and three-class packets.
3. Continuous parameter bounds in [0.0, 1.0].
4. Quality flag behavior and command gating.
5. Backward compatibility between legacy binary packets and 3-class Unity HandController.cs.
"""

import json
import math
import socket
import time
import pytest


UDP_HOST = "127.0.0.1"
UDP_PORT = 5005


def simulate_unity_hand_controller(packet_dict, current_time=0.0, last_received_time=0.0, timeout_s=0.75):
    """Simulate the deserialization and target computation in Unity HandController.cs."""
    if current_time - last_received_time > timeout_s:
        return {"target_flex": 0.0, "target_splay": 0.0}

    # JsonUtility defaults for missing fields in C#
    grasp_cmd = int(packet_dict.get("grasp_command", 0))
    ext_cmd = int(packet_dict.get("extension_command", 0))
    cont_flex = float(packet_dict.get("continuous_flex", 0.0))
    cont_splay = float(packet_dict.get("continuous_splay", 0.0))
    class_name = packet_dict.get("class_name", "")
    quality = packet_dict.get("quality", "ok")

    is_ok = (quality == "ok" or quality == "")

    target_flex = 0.0
    if not math.isnan(cont_flex) and not math.isinf(cont_flex):
        grasp_active = (grasp_cmd == 1 or
                        class_name in ("grasp", "volar_grasp", "right_hand_grasp"))
        target_flex = max(0.0, min(1.0, cont_flex)) if (grasp_active and is_ok) else 0.0

    target_splay = 0.0
    if not math.isnan(cont_splay) and not math.isinf(cont_splay):
        ext_active = (ext_cmd == 1 or
                      class_name in ("extension", "finger_extension"))
        target_splay = max(0.0, min(1.0, cont_splay)) if (ext_active and is_ok) else 0.0

    return {"target_flex": target_flex, "target_splay": target_splay}


@pytest.fixture
def udp_receiver_5005():
    """Create and tear down a UDP receiver socket bound to 127.0.0.1:5005."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(2.0)
    try:
        yield sock
    finally:
        sock.close()


def test_udp_socket_send_and_receive_binary_packet(udp_receiver_5005):
    """Verify raw binary packet serialization, UDP transmission on port 5005, and receipt."""
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        binary_payload = {
            "timestamp": time.time(),
            "sample_end": 1024,
            "p_raw": 0.85,
            "p_smooth": 0.82,
            "grasp_command": 1,
            "continuous_flex": 0.64,
            "state_label": "ACTION",
            "quality": "ok",
            "threshold": 0.5,
        }
        data = json.dumps(binary_payload, allow_nan=False).encode("utf-8")
        sender.sendto(data, (UDP_HOST, UDP_PORT))

        recv_bytes, addr = udp_receiver_5005.recvfrom(4096)
        assert addr[0] == "127.0.0.1"

        received = json.loads(recv_bytes.decode("utf-8"))
        assert received["grasp_command"] == 1
        assert received["continuous_flex"] == pytest.approx(0.64, rel=1e-4)
        assert received["quality"] == "ok"
        assert received["state_label"] == "ACTION"
        assert 0.0 <= received["continuous_flex"] <= 1.0

        # Simulate Unity HandController parsing
        unity_out = simulate_unity_hand_controller(received)
        assert unity_out["target_flex"] == pytest.approx(0.64, rel=1e-4)
        assert unity_out["target_splay"] == pytest.approx(0.0)
    finally:
        sender.close()


def test_udp_socket_send_and_receive_threeclass_packets(udp_receiver_5005):
    """Verify 3-class UDP packets (rest, grasp, extension) on port 5005."""
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 1. Rest packet
        rest_payload = {
            "timestamp": time.time(),
            "class_id": 0,
            "class_name": "rest",
            "grasp_command": 0,
            "extension_command": 0,
            "continuous_flex": 0.0,
            "continuous_splay": 0.0,
            "probabilities": [0.90, 0.06, 0.04],
            "quality": "ok",
        }
        sender.sendto(json.dumps(rest_payload).encode("utf-8"), (UDP_HOST, UDP_PORT))
        data, _ = udp_receiver_5005.recvfrom(4096)
        parsed_rest = json.loads(data.decode("utf-8"))
        assert parsed_rest["class_name"] == "rest"
        assert parsed_rest["grasp_command"] == 0
        assert parsed_rest["extension_command"] == 0
        assert parsed_rest["continuous_flex"] == 0.0
        assert parsed_rest["continuous_splay"] == 0.0
        out_rest = simulate_unity_hand_controller(parsed_rest)
        assert out_rest["target_flex"] == 0.0
        assert out_rest["target_splay"] == 0.0

        # 2. Grasp packet (volar_grasp)
        grasp_payload = {
            "timestamp": time.time(),
            "class_id": 1,
            "class_name": "volar_grasp",
            "grasp_command": 1,
            "extension_command": 0,
            "continuous_flex": 0.78,
            "continuous_splay": 0.0,
            "probabilities": [0.10, 0.78, 0.12],
            "quality": "ok",
        }
        sender.sendto(json.dumps(grasp_payload).encode("utf-8"), (UDP_HOST, UDP_PORT))
        data, _ = udp_receiver_5005.recvfrom(4096)
        parsed_grasp = json.loads(data.decode("utf-8"))
        assert parsed_grasp["class_name"] == "volar_grasp"
        assert parsed_grasp["grasp_command"] == 1
        assert parsed_grasp["extension_command"] == 0
        assert parsed_grasp["continuous_flex"] == pytest.approx(0.78, rel=1e-4)
        out_grasp = simulate_unity_hand_controller(parsed_grasp)
        assert out_grasp["target_flex"] == pytest.approx(0.78, rel=1e-4)
        assert out_grasp["target_splay"] == 0.0

        # 3. Extension packet (finger_extension)
        ext_payload = {
            "timestamp": time.time(),
            "class_id": 2,
            "class_name": "finger_extension",
            "grasp_command": 0,
            "extension_command": 1,
            "continuous_flex": 0.0,
            "continuous_splay": 0.85,
            "probabilities": [0.05, 0.10, 0.85],
            "quality": "ok",
        }
        sender.sendto(json.dumps(ext_payload).encode("utf-8"), (UDP_HOST, UDP_PORT))
        data, _ = udp_receiver_5005.recvfrom(4096)
        parsed_ext = json.loads(data.decode("utf-8"))
        assert parsed_ext["class_name"] == "finger_extension"
        assert parsed_ext["grasp_command"] == 0
        assert parsed_ext["extension_command"] == 1
        assert parsed_ext["continuous_splay"] == pytest.approx(0.85, rel=1e-4)
        out_ext = simulate_unity_hand_controller(parsed_ext)
        assert out_ext["target_flex"] == 0.0
        assert out_ext["target_splay"] == pytest.approx(0.85, rel=1e-4)
    finally:
        sender.close()


def test_continuous_ranges_and_clamping():
    """Verify that continuous flex and splay values clamp cleanly to [0.0, 1.0]."""
    test_cases = [
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.35, 0.35),
        (1.0, 1.0),
        (1.5, 1.0),
    ]

    for input_val, expected in test_cases:
        packet_grasp = {
            "grasp_command": 1,
            "extension_command": 0,
            "continuous_flex": input_val,
            "continuous_splay": 0.0,
            "quality": "ok",
        }
        res_grasp = simulate_unity_hand_controller(packet_grasp)
        assert res_grasp["target_flex"] == pytest.approx(expected, rel=1e-5)
        assert 0.0 <= res_grasp["target_flex"] <= 1.0

        packet_ext = {
            "grasp_command": 0,
            "extension_command": 1,
            "continuous_flex": 0.0,
            "continuous_splay": input_val,
            "quality": "ok",
        }
        res_ext = simulate_unity_hand_controller(packet_ext)
        assert res_ext["target_splay"] == pytest.approx(expected, rel=1e-5)
        assert 0.0 <= res_ext["target_splay"] <= 1.0


def test_quality_flags_gating():
    """Verify that non-'ok' quality states force target actuation to 0.0."""
    non_ok_qualities = ["warming_up", "paused", "calibrating", "shutdown", "saved", "error"]

    for q in non_ok_qualities:
        # Grasp packet with non-ok quality
        packet = {
            "grasp_command": 1,
            "extension_command": 0,
            "continuous_flex": 0.9,
            "continuous_splay": 0.0,
            "quality": q,
        }
        res = simulate_unity_hand_controller(packet)
        assert res["target_flex"] == 0.0, f"Expected 0.0 for quality='{q}', got {res['target_flex']}"
        assert res["target_splay"] == 0.0

        # Extension packet with non-ok quality
        packet_ext = {
            "grasp_command": 0,
            "extension_command": 1,
            "continuous_flex": 0.0,
            "continuous_splay": 0.9,
            "quality": q,
        }
        res_ext = simulate_unity_hand_controller(packet_ext)
        assert res_ext["target_flex"] == 0.0
        assert res_ext["target_splay"] == 0.0, f"Expected 0.0 for quality='{q}', got {res_ext['target_splay']}"

    # Verify 'ok' quality enables actuation
    ok_packet = {
        "grasp_command": 1,
        "extension_command": 0,
        "continuous_flex": 0.75,
        "quality": "ok",
    }
    res_ok = simulate_unity_hand_controller(ok_packet)
    assert res_ok["target_flex"] == pytest.approx(0.75, rel=1e-4)


def test_packet_backward_compatibility():
    """Verify legacy binary packet compatibility with 3-class receiver."""
    legacy_packet = {
        "timestamp": 12345.67,
        "sample_end": 500,
        "p_raw": 0.9,
        "p_smooth": 0.88,
        "grasp_command": 1,
        "continuous_flex": 0.76,
        "state_label": "ACTION",
        "quality": "ok",
        "threshold": 0.5,
    }

    # 1. Confirm legacy packet lacks 3-class fields
    assert "extension_command" not in legacy_packet
    assert "continuous_splay" not in legacy_packet
    assert "class_name" not in legacy_packet

    # 2. Deserializing into 3-class Unity HandController defaults missing fields cleanly
    out = simulate_unity_hand_controller(legacy_packet)
    assert out["target_flex"] == pytest.approx(0.76, rel=1e-4)
    assert out["target_splay"] == 0.0

    # 3. Legacy resting packet
    legacy_rest = {
        "timestamp": 12346.0,
        "sample_end": 550,
        "p_raw": 0.1,
        "p_smooth": 0.12,
        "grasp_command": 0,
        "continuous_flex": 0.0,
        "state_label": "REST",
        "quality": "ok",
        "threshold": 0.5,
    }
    out_rest = simulate_unity_hand_controller(legacy_rest)
    assert out_rest["target_flex"] == 0.0
    assert out_rest["target_splay"] == 0.0


def test_timeout_and_rest_fallback():
    """Verify that timeout (exceeding timeout_s) resets targets to 0.0."""
    packet = {
        "grasp_command": 1,
        "continuous_flex": 0.8,
        "quality": "ok",
    }
    # Within timeout
    active = simulate_unity_hand_controller(packet, current_time=1.0, last_received_time=0.5, timeout_s=0.75)
    assert active["target_flex"] == pytest.approx(0.8, rel=1e-4)

    # Exceeding timeout (0.8s elapsed > 0.75s timeout)
    timed_out = simulate_unity_hand_controller(packet, current_time=1.35, last_received_time=0.5, timeout_s=0.75)
    assert timed_out["target_flex"] == 0.0
    assert timed_out["target_splay"] == 0.0


def test_nan_infinity_sanitization():
    """Verify that NaN or Infinity in continuous parameters is handled without error."""
    nan_packet = {
        "grasp_command": 1,
        "extension_command": 1,
        "continuous_flex": float("nan"),
        "continuous_splay": float("inf"),
        "quality": "ok",
    }
    out = simulate_unity_hand_controller(nan_packet)
    assert out["target_flex"] == 0.0
    assert out["target_splay"] == 0.0


def test_studio_expanded_payload_unity_mapping(udp_receiver_5005):
    """Verify that the full multi-class continuous payload from studio servers maps cleanly to Unity HandController."""
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Full studio payload format with probabilities and class telemetry
        full_payload = {
            "timestamp": time.time(),
            "sample_end": 5000,
            "class_name": "finger_extension",
            "class_id": 2,
            "grasp_command": 0,
            "extension_command": 1,
            "continuous_flex": 0.0,
            "continuous_splay": 0.72,
            "probabilities": {
                "rest": 0.08,
                "volar_grasp": 0.05,
                "finger_extension": 0.87,
            },
            "probabilities_raw": [0.08, 0.05, 0.87],
            "probabilities_ema": [0.10, 0.07, 0.83],
            "quality": "ok",
            "p_raw": 0.87,
            "p_smooth": 0.83,
            "state_label": "finger_extension",
            "threshold": 0.5,
        }

        # Transmit over UDP 5005
        sender.sendto(json.dumps(full_payload).encode("utf-8"), (UDP_HOST, UDP_PORT))
        raw_bytes, addr = udp_receiver_5005.recvfrom(8192)
        received = json.loads(raw_bytes.decode("utf-8"))

        # Verify all studio fields arrived
        assert received["class_name"] == "finger_extension"
        assert received["class_id"] == 2
        assert received["grasp_command"] == 0
        assert received["extension_command"] == 1
        assert received["continuous_flex"] == 0.0
        assert received["continuous_splay"] == pytest.approx(0.72, rel=1e-4)
        assert isinstance(received["probabilities"], dict)
        assert received["quality"] == "ok"

        # Verify Unity mapping:
        unity = simulate_unity_hand_controller(received)
        assert unity["target_flex"] == 0.0
        assert unity["target_splay"] == pytest.approx(0.72, rel=1e-4)
    finally:
        sender.close()

