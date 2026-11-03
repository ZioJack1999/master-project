using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

// Reference Unity receiver. Assign an Animator with a float parameter Flexion.
// UDP transport is tested by the Python suite; this script needs a lab Unity build.
public sealed class HandController : MonoBehaviour
{
    public Animator handAnimator;
    public int port = 5005;
    public float timeoutSeconds = 0.75f;
    private UdpClient client;
    private Thread thread;
    private volatile bool running;
    private readonly object gate = new object();
    private string latest;
    private float receivedAt = -1000f;
    private float targetFlex;
    private float currentFlex;
    [Serializable] public class Packet
    {
        public int grasp_command;
        public float continuous_flex;
        public string quality;
    }
    void OnEnable()
    {
        lock (gate) latest = null;
        receivedAt = -1000f;
        targetFlex = currentFlex = 0f;
        client = new UdpClient(new IPEndPoint(IPAddress.Loopback, port));
        running = true;
        thread = new Thread(Receive) { IsBackground = true };
        thread.Start();
    }
    private void Receive()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try {
                byte[] data = client.Receive(ref remote);
                lock (gate) latest = Encoding.UTF8.GetString(data);
            }
            catch (SocketException) { if (!running) break; }
            catch (ObjectDisposedException) { break; }
        }
    }
    void Update()
    {
        string text;
        lock (gate) { text = latest; latest = null; }
        if (text != null) {
            try {
                Packet p = JsonUtility.FromJson<Packet>(text);
                if (p != null && !float.IsNaN(p.continuous_flex) && !float.IsInfinity(p.continuous_flex)) {
                    targetFlex = p.grasp_command == 1 && p.quality == "ok" ? Mathf.Clamp01(p.continuous_flex) : 0f;
                    receivedAt = Time.realtimeSinceStartup;
                }
            }
            catch (ArgumentException) { targetFlex = 0f; }
        }
        if (Time.realtimeSinceStartup - receivedAt > timeoutSeconds) targetFlex = 0f;
        currentFlex = Mathf.Lerp(currentFlex, targetFlex, 1f - Mathf.Exp(-10f * Time.deltaTime));
        if (handAnimator != null) handAnimator.SetFloat("Flexion", currentFlex);
    }
    void OnDisable()
    {
        running = false;
        client?.Close();
        thread?.Join(1000);
        targetFlex = currentFlex = 0f;
        if (handAnimator != null) handAnimator.SetFloat("Flexion", 0f);
    }
}
