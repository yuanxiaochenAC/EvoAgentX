import os
import subprocess
import threading
import time
import signal
import atexit
import json
from datetime import datetime
from .api import app
from dotenv import load_dotenv

load_dotenv()
target_port = 8001
CREATE_PUBLIC_TUNNEL = True
TUNNEL_INFO_PATH = "./server/tunnel_info.json"


# Global variable to track the tunnel process
tunnel_process = None

def cleanup_tunnel():
    """Clean up the tunnel process"""
    global tunnel_process
    if tunnel_process and tunnel_process.poll() is None:
        print("🧹 Cleaning up tunnel process...")
        try:
            tunnel_process.terminate()
            # Give it a moment to terminate gracefully
            tunnel_process.wait(timeout=5)
            print("✅ Tunnel process terminated")
        except subprocess.TimeoutExpired:
            print("⚠️  Tunnel process didn't terminate gracefully, forcing kill...")
            tunnel_process.kill()
            tunnel_process.wait()
            print("✅ Tunnel process killed")
        except Exception as e:
            print(f"❌ Error cleaning up tunnel: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    cleanup_tunnel()
    exit(0)

def save_tunnel_info(url, port):
    """Save tunnel information to JSON file"""
    tunnel_info = {
        "public_url": url,
        "local_port": port,
        "local_url": f"http://localhost:{port}",
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }
    
    try:
        with open(TUNNEL_INFO_PATH, "w") as f:
            json.dump(tunnel_info, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Failed to save tunnel info: {e}")
        return False

def create_public_tunnel(port):
    """Create a public tunnel using localhost.run"""
    global tunnel_process
    try:
        # Run the SSH tunnel command silently
        tunnel_process = subprocess.Popen(
            ["ssh", "-R", f"80:localhost:{port}", "localhost.run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read the output to get the public URL
        url_found = False
        qr_section = False
        
        for line in tunnel_process.stdout:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Detect QR code section and skip it
            if "Open your tunnel address on your mobile with this QR:" in line:
                qr_section = True
                continue
            elif qr_section and line and not line.startswith("http"):
                # Skip QR code lines
                continue
            elif qr_section and line.startswith("http"):
                # End of QR section
                qr_section = False
            
            # Look for the tunnel URL
            if "tunneled with tls termination" in line:
                # Extract and save the URL silently
                import re
                url_match = re.search(r'https?://[^\s]+\.(localhost\.run|lhr\.life)', line)
                if url_match:
                    public_url = url_match.group(0)
                    # Save to JSON file silently
                    save_tunnel_info(public_url, port)
                    url_found = True
                    break
        
        # Only print if there's an issue
        if not url_found:
            print("⚠️  Public URL not detected, tunnel may still be connecting...")
                
        # Keep the process running
        tunnel_process.wait()
        
    except Exception as e:
        print(f"❌ Failed to create public tunnel: {e}")
        print("💡 Make sure SSH is installed and you have internet connection")

if __name__ == "__main__":
    import uvicorn
    
    # Register cleanup function to run on exit
    atexit.register(cleanup_tunnel)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Check if public tunnel should be created
    create_tunnel = CREATE_PUBLIC_TUNNEL
    
    if create_tunnel:
        # Start the tunnel in a separate thread
        tunnel_thread = threading.Thread(target=create_public_tunnel, args=(target_port,))
        tunnel_thread.daemon = True
        tunnel_thread.start()
        
        # Give the tunnel a moment to start
        time.sleep(2)
    
    print(f"🖥️  Local server: http://localhost:{target_port}")
    if create_tunnel:
        print("🌍 Public tunnel: URL will be saved to tunnel_info.json")
    else:
        tunnel_info = {
            "public_url": None,
            "local_port": target_port,
            "local_url": f"http://localhost:{target_port}",
            "created_at": datetime.now().isoformat(),
            "status": "inactive"
        }
        with open(TUNNEL_INFO_PATH, "w") as f:
            json.dump(tunnel_info, f, indent=2)
        print("🌍 Public tunnel: Not created")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=target_port)
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    finally:
        cleanup_tunnel()


