<#
.SYNOPSIS
    Expose the local MC-MOT server over HTTPS so a phone browser can use its camera.

.DESCRIPTION
    Runs a Cloudflare quick tunnel in front of the local server and prints the
    https:// URL to open on the phone.

    Why a tunnel at all, for a system that runs entirely on your own LAN: browsers
    only expose getUserMedia() in a *secure context*. That means localhost or
    HTTPS — and http://192.168.1.x is neither. Without HTTPS the Registration page
    physically cannot open a phone camera, whatever permissions are granted.

    Note the asymmetry, because it decides what you actually need this for:

      Registration  the phone's *browser* captures    -> needs HTTPS (this script)
      Tracking      the server *pulls* a phone stream -> plain LAN HTTP is fine

    So tracking a phone camera through an IP-camera app works without any of this.
    Only browser capture needs the tunnel.

    Alternatives considered, and why this one:

      Cloudflare Tunnel  chosen. No account for a quick tunnel, real certificate,
                         one binary, works on cellular as well as Wi-Fi. Costs a
                         round trip through Cloudflare (~30-80 ms) and the video
                         leaves your network.
      Tailscale          lowest latency and stays on your own network, and it can
                         issue real certificates via MagicDNS + HTTPS. Needs a
                         client on both devices and an account. The better answer
                         if you use it already — see docs/mobile.md.
      ngrok              equivalent to Cloudflare but rate-limits free tunnels and
                         shows an interstitial page that breaks an <img> MJPEG tag.
      LocalTunnel        no account, but frequently unreachable and slow; fine for
                         a one-off demo, not for repeated use.
      Self-signed cert   stays on the LAN with no third party, but Android requires
                         installing a CA into the user trust store, and Chrome
                         still refuses some APIs. Fragile in practice.

.PARAMETER Port
    Local port the MC-MOT server is listening on. Defaults to 8000.

.PARAMETER Install
    Download cloudflared into .tools\ if it is not already available.

.EXAMPLE
    .\scripts\tunnel.ps1 -Install
    .\scripts\tunnel.ps1 -Port 8000
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$toolsDir = Join-Path $root '.tools'
$localExe = Join-Path $toolsDir 'cloudflared.exe'

# ---------------------------------------------------------------- locate binary
function Resolve-Cloudflared {
    $onPath = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    if (Test-Path $localExe) { return $localExe }
    return $null
}

$exe = Resolve-Cloudflared

if (-not $exe -and $Install) {
    Write-Host '==> Downloading cloudflared' -ForegroundColor Cyan
    if (-not (Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir | Out-Null }
    $url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
    # To a temporary name first, then renamed: an interrupted download must not
    # leave a truncated executable behind that looks installed.
    $partial = "$localExe.partial"
    Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing
    Move-Item -Force $partial $localExe
    $exe = $localExe
    Write-Host "    installed to $localExe" -ForegroundColor DarkGray
}

if (-not $exe) {
    Write-Host 'cloudflared was not found.' -ForegroundColor Red
    Write-Host 'Run this script again with -Install, or install it yourself:' -ForegroundColor Yellow
    Write-Host '    winget install --id Cloudflare.cloudflared' -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------- check server
$listening = $null
try {
    $listening = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/system/version" `
        -UseBasicParsing -TimeoutSec 3
}
catch {
    $listening = $null
}

if (-not $listening) {
    # A warning, not a failure: starting the tunnel first is a legitimate order,
    # and cloudflared will happily wait for the origin to appear.
    Write-Host "!! Nothing is answering on http://127.0.0.1:$Port" -ForegroundColor Yellow
    Write-Host '   Start the server in another terminal:' -ForegroundColor Yellow
    Write-Host '       .venv\Scripts\python.exe -m mcmot.main' -ForegroundColor Yellow
    Write-Host '   The tunnel will connect once it is up.' -ForegroundColor DarkGray
}
else {
    $version = ($listening.Content | ConvertFrom-Json).version
    Write-Host "==> MC-MOT v$version is listening on port $Port" -ForegroundColor Green
}

Write-Host ''
Write-Host '==> Starting Cloudflare quick tunnel' -ForegroundColor Cyan
Write-Host '    Watch for the https://<random>.trycloudflare.com URL below,' -ForegroundColor DarkGray
Write-Host '    then open it on your phone and go to Registration.' -ForegroundColor DarkGray
Write-Host '    The URL changes every run — a named tunnel is the fix for that;' -ForegroundColor DarkGray
Write-Host '    see docs/mobile.md.' -ForegroundColor DarkGray
Write-Host ''

# --no-autoupdate: an update mid-session would restart the tunnel and change the
# URL, which is exactly what you do not want while someone is holding a phone.
& $exe tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
