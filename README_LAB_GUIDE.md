# Real-Time BCI Lab Operational Manual & Fast Calibration Guide
========================================================================

This manual provides complete step-by-step instructions for operating the standalone real-time motor imagery decoder in the laboratory with physical electrodes and a virtual hand.

---

## 1. Hardware & Electrode Preparation

### 1.1 Electrode Montage & Target Sites
Ensure your 20-channel EEG cap aligns with the 10–20 international system:
* **Primary Motor Channels (Crucial)**: `C3` (Left Hemisphere / Right Hand area), `Cz` (Vertex / Foot/Hand midline), `C4` (Right Hemisphere).
* **Frontal & Parieto-Occipital Channels**: `F3`, `F1`, `Fz`, `F2`, `F4`, `C1`, `C2`, `CPz`, `P3`, `P1`, `Pz`, `P2`, `P4`, `POz`, `O1`, `Oz`, `O2`.
* **Reference & Ground**: Earlobe (`A1/A2`) or Mastoid (`M1/M2`), Ground at `AFz` or `Fpz`.

### 1.2 Impedance Check
* Inject conductive gel into each electrode well.
* Target impedance: **$< 10\text{ k}\Omega$** across all motor cortex channels (`C3`, `Cz`, `C4`).
* Verify reference and ground channels have clean, low-impedance contact.

---

## 2. Launching the Electrode Stream (LabStreamingLayer)

1. Open your acquisition software (e.g. OpenBCI GUI, BrainVision RDA / LSL Connector, g.tec Nautilus, ANT Neuro eego, LabRecorder).
2. Configure stream parameters:
   * **Stream Type**: `EEG`
   * **Sampling Rate**: `256 Hz` (or allow internal resampling if native rate is 500/1000 Hz)
   * **Channel Count**: 20 channels (matching the standard montage)
3. Click **Start Streaming** / **LSL Broadcast**.

---

## 3. The 60-Second Guided EEG Calibration Routine

Because every user has a unique skull thickness, baseline $\mu/\beta$ band power, and resting voltage offset, **always run the 60-second calibration when putting on the cap**:

### 3.1 Start Calibration Command
Open PowerShell or Command Prompt in the `multi_nn` conda environment:
```powershell
& "C:\tools\Anaconda3\envs\multi_nn\python.exe" C:\Users\giaco\Desktop\BCI_EEGMI_Results_and_Decoder\decoder_and_models\realtime_bci_decoder.py --mode calibrate --udp-port 5005
```

### 3.2 What Happens During Calibration
The script guides you through **3 alternating pairs** of Rest and Motor Imagery (5.0 seconds per trial, 60 seconds total):

```
[Trial 1/3] ---> RELAX / REST (5s) <---
  • Sit comfortably, look at the screen center.
  • Keep jaw, neck, and facial muscles completely relaxed.
  • Do not imagine any hand movement.

[Trial 1/3] >>> IMAGINE RIGHT HAND GRASP / CLOSING (5s) <<<
  • Perform kinesthetic motor imagery of closing your right hand into a fist.
  • Feel the muscle tension in your mind (do not physically move fingers).

[Trial 2/3] ---> RELAX / REST (5s) <---
[Trial 2/3] >>> IMAGINE RIGHT HAND GRASP / CLOSING (5s) <<<

[Trial 3/3] ---> RELAX / REST (5s) <---
[Trial 3/3] >>> IMAGINE RIGHT HAND GRASP / CLOSING (5s) <<<
```

### 3.3 Automatic Calibration Computation
At the end of 60 seconds, the decoder automatically:
1. Calculates your personal baseline mean ($\mu$) and standard deviation ($\sigma$).
2. Sweeps decision thresholds via ROC analysis.
3. Prints your optimal threshold:
   ```
   [Calibration Complete] Computing optimal user threshold...
     Calibrated Threshold: 0.32 | Calibration Balanced Acc: 86.4%
   ```
4. **Instantly and seamlessly transitions into live decoding mode.**

---

## 4. Live Real-Time Closed-Loop Control

Once calibrated, the terminal displays real-time telemetry updating at **4.0 Hz (every 250 ms)**:

```
[LIVE] Step: 00142 | Latency:  5.1ms | p(EMA): 0.78 | Hand: ACTION [████████░░]
```

### Signal Processing Safeguards in Real Time:
* **Online Bandpass (8–30 Hz)**: 4th-order Butterworth with persistent `zi` state to prevent edge blips.
* **Artifact Clamping**: Soft-clips voltage spikes $> \pm 100\ \mu\text{V}$ from eye blinks or swallowing.
* **EMA Smoothing ($\alpha = 0.3$)**: Eliminates jerky, sporadic probability spikes.
* **Dwell-Time Hysteresis ($500\text{ ms}$)**: Requires sustained intent before triggering hand closure.

---

## 5. Virtual Hand Integration (UDP Port 5005)

Every 250 ms, the decoder broadcasts a JSON datagram to `127.0.0.1:5005`:
```json
{
  "timestamp": 1771781630.12,
  "p_raw": 0.8421,
  "p_smooth": 0.7610,
  "grasp_command": 1,
  "continuous_flex": 0.8850,
  "state_label": "ACTION"
}
```

### 5.1 Unity C# Receiver Script (`HandController.cs`)
Attach this script to your Virtual Hand GameObject in Unity:

```csharp
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class HandController : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    public Animator handAnimator;
    public float currentFlex = 0f;
    private float targetFlex = 0f;

    [System.Serializable]
    public class BciPacket
    {
        public float p_smooth;
        public int grasp_command;
        public float continuous_flex;
        public string state_label;
    }

    void Start()
    {
        udpClient = new UdpClient(5005);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string text = Encoding.UTF8.GetString(data);
                BciPacket packet = JsonUtility.FromJson<BciPacket>(text);
                targetFlex = packet.continuous_flex;
            }
            catch {}
        }
    }

    void Update()
    {
        // Smooth interpolation for fluid finger animation
        currentFlex = Mathf.Lerp(currentFlex, targetFlex, Time.deltaTime * 10f);
        if (handAnimator != null)
        {
            handAnimator.SetFloat("Flexion", currentFlex);
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (udpClient != null) udpClient.Close();
    }
}
```

### 5.2 Standalone Python Virtual Hand Tester
If you want to test the UDP receiver without Unity:
```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))
print("Listening for BCI Virtual Hand packets on port 5005...")

while True:
    data, _ = sock.recvfrom(1024)
    packet = json.loads(data.decode('utf-8'))
    bar = "█" * int(packet['continuous_flex'] * 20)
    print(f"\rHand State: {packet['state_label']:<6} | Flex: {packet['continuous_flex']:.2f} [{bar:<20}]", end="", flush=True)
```

---

## 6. Troubleshooting Common Lab Issues

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **No LSL stream detected** | Amplifier software not broadcasting LSL | Verify `type='EEG'` and start LSL stream before running decoder. |
| **Hand triggers on rest** | Baseline threshold too low or subject tense | Re-run `--mode calibrate` and ensure jaw/shoulders are completely relaxed. |
| **Hand slow to respond** | Dwell-time too conservative | Decrease dwell steps (e.g. `dwell_steps=1` in `realtime_bci_decoder.py`). |
| **High 50/60 Hz noise** | Mains line interference | Ensure ground electrode impedance is $< 5\text{ k}\Omega$; 8–30 Hz filter will reject mains noise. |
