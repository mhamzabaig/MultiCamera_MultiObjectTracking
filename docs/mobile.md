# Using your phone

Two completely different things get called "using my phone as a camera", and they
have different requirements. Getting this distinction right saves a lot of wasted
effort:

| | What the phone does | Who connects to whom | Needs HTTPS? |
|---|---|---|---|
| **Registration** | Its *browser* captures frames | Phone → laptop | **Yes** |
| **Tracking** | An app serves a video stream | Laptop → phone | **No** |

The reason is `getUserMedia()`. Browsers only expose a camera in a **secure
context** — `localhost` or HTTPS. `http://192.168.1.42:8000` is neither, so the
Registration page physically cannot open the camera there, no matter what
permissions you grant. Tracking has no such problem because the *server* dials out
to the phone.

So if all you want is to track a phone camera, skip to
[Tracking from a phone](#tracking-from-a-phone) — no tunnel, no certificate, nothing
to install on the laptop.

## Tracking from a phone

Plain LAN, both devices on the same Wi-Fi.

1. Install an IP-camera app on the phone. **IP Webcam** (Android) is the usual
   choice; anything that serves MJPEG over HTTP or RTSP works.
2. Start the server in the app. It shows a URL such as `http://192.168.1.42:8080`.
3. On the laptop, open the MC-MOT **Cameras** page and add:

   | Field | Value |
   |---|---|
   | Camera id | `phone` |
   | Source | `http://192.168.1.42:8080/video` |

   IP Webcam serves MJPEG at `/video`. For RTSP apps, use the `rtsp://…` URL they
   print. Credentials in the URL are stripped before it is stored or displayed.
4. The camera appears as `connecting`, then `streaming`. If the phone is not ready
   yet that is fine — it reconnects on its own with backoff.
5. Go to **Tracking**, select `phone`, and press Start.

Keep the phone on mains power and disable screen timeout; most IP-camera apps stop
streaming when the screen locks.

## Registration from a phone

This needs HTTPS, which means a tunnel.

```powershell
# once, to fetch the binary into .tools\
.\scripts\tunnel.ps1 -Install

# each session
.\scripts\tunnel.ps1 -Port 8000
```

The script prints an `https://<random>.trycloudflare.com` URL. Open it on the phone,
go to **Registration**, press **Enable camera**, and allow the permission prompt.

Two things worth knowing:

* **The URL changes every run.** Quick tunnels are anonymous and ephemeral. If you
  will do this often, a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
  gives a stable hostname, at the cost of a Cloudflare account and a domain.
* **Video goes through Cloudflare.** Both the registration upload and any MJPEG you
  watch over that URL leave your network. For registration that is a handful of
  JPEGs; for continuous tracking it is real bandwidth, and the LAN address is the
  better way to watch.

### Access methods compared

| Method | Latency | Bandwidth | Security | Setup | Limits |
|---|---|---|---|---|---|
| **Cloudflare Tunnel** *(chosen)* | +30–80 ms | Through Cloudflare | Real cert; traffic transits a third party | One binary, no account | Ephemeral URL unless named |
| Tailscale | +2–10 ms, stays on your network | Direct, usually LAN-speed | WireGuard, end to end | Client on both devices, free account | Needs MagicDNS + HTTPS for a cert |
| ngrok | +40–100 ms | Through ngrok | Real cert | Account required | Free tier rate-limits, and its interstitial page breaks an `<img>` MJPEG tag |
| LocalTunnel | Variable, often poor | Through their relay | Real cert | None | Frequently unreachable; fine for a one-off |
| Plain LAN HTTP | Best | LAN only | None, but never leaves the network | None | **`getUserMedia` blocked — registration impossible** |
| Self-signed cert | Best | LAN only | Only as trusted as your CA | Install a CA on Android | Fragile; Chrome still refuses some APIs |

**Cloudflare Tunnel is the default** because it works with one downloaded binary and
no account, on Wi-Fi or cellular. **Tailscale is the better answer if you already use
it** — lower latency, and the video never leaves your network. Both are a config
choice, not a redesign.

A practical hybrid, and what I would actually do day to day: **register over the
tunnel, watch tracking over the LAN address.** Registration is a few seconds of a few
hundred kilobytes; tracking is a continuous stream that has no business crossing the
internet to reach a laptop in the same room.

## Calibrating the thresholds

This matters more than it sounds, and it is the step most likely to decide whether
the system works for you.

The shipped `tau_accept` of 0.55 was measured on crops from **one photo**, augmented
to simulate a second camera. Your real case is harder: enrolment crops come from a
phone held at chest height, tracking crops come from a different camera at a
different angle in different light. Published ReID benchmark numbers do not transfer
to that, and the honest position is that the right threshold for *your* cameras is
unknown until measured.

Collect images per person from both cameras:

```
calibration/
  hamza/
    enrol/     20-30 frames from the phone, as you would register
    verify/    20-30 frames of the same person from the tracking camera
  ali/
    enrol/
    verify/
```

Two people minimum. The second person is what supplies the *strangers* — every other
person's `verify` images are strangers to this person's gallery, which is how false
accepts get measured for free.

```powershell
.venv\Scripts\python.exe scripts\calibrate.py calibration\
```

It reports rank-1 accuracy, the score distributions for same-person and
different-person pairs, a sweep over both thresholds, and a recommendation:

```
same person   (n= 24) : min 0.968  mean 0.974  max 0.979
different     (n= 24) : min 0.461  mean 0.494  max 0.511
separation             : +0.481

    identity:
      tau_accept: 0.75
      tau_margin: 0.0
```

Put the recommendation in `config/local.yaml` and restart.

**How to read the output.** The recommendation is not the threshold with the highest
recall — when the two distributions separate cleanly, *every* low threshold scores
100% with no false accepts, and choosing the lowest of those would leave no headroom
for a stranger who happens to score higher than anyone in your sample. It picks the
midpoint of the observed gap instead: the point furthest from both failure modes.

**If no threshold eliminates false accepts**, the two views are too different for the
model to bridge. In order of effectiveness:

1. Enrol from the camera you will actually track with, or from both.
2. Add frames covering more angles and distances — a gallery of near-identical frames
   looks confident but covers one viewpoint.
3. Improve the tracking camera's framing: full body, well lit, not heavily occluded.
4. Only then consider a heavier ReID backbone. `BaseEmbedder` allows a second
   backend without touching the pipeline.

The asymmetry to keep in mind while tuning: a missed match costs a few frames of
"Unknown", while a false accept puts the wrong name on a person's head. Prefer
Unknown.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Registration says camera access needs HTTPS | You are on `http://<lan-ip>`. Use the tunnel URL. |
| Permission prompt never appears | Already denied for this origin. Clear site settings for the tunnel host, or use a fresh URL. |
| Camera stuck at `connecting` | Phone app not started, wrong path (`/video` for IP Webcam), or a different subnet. Open the URL in the laptop's browser to check. |
| Camera goes to `reconnecting` repeatedly | Phone screen locked, or the app stopped. Disable screen timeout. |
| Stream shows a frozen picture | The pipeline was stopped. The stream ends after a few seconds of no frames; reload the page. |
| Everyone shows as `Unknown` | Nobody enrolled, or `tau_accept` is too high for your cameras. Calibrate. |
| Two people share one name | The gallery mixed two people during registration — someone walked behind you and became the largest detection. Delete and re-register alone. |
| `state=ended` on a camera | A video-file source reached its end. Files do not loop. |
