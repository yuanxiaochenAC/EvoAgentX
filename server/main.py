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
    """Clean up the tunnel process silently"""
    global tunnel_process
    if tunnel_process and tunnel_process.poll() is None:
        try:
            tunnel_process.terminate()
            tunnel_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            tunnel_process.kill()
            tunnel_process.wait()
        except Exception:
            pass  # Suppress cleanup errors

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
    except Exception:
        return False

def create_public_tunnel(port):
    """Create a public tunnel using localhost.run - completely silent"""
    global tunnel_process
    try:
        # Run the SSH tunnel command with all output suppressed
        tunnel_process = subprocess.Popen(
            ["ssh", "-R", f"80:localhost:{port}", "localhost.run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Suppress stderr completely
            stdin=subprocess.DEVNULL,   # Suppress stdin
            text=True
        )
        
        # Read output silently to extract URL
        try:
            # Give it time to establish connection
            time.sleep(3)
            
            # Try to read some output to get the URL
            output_lines = []
            while len(output_lines) < 20:  # Read first 20 lines max
                try:
                    line = tunnel_process.stdout.readline()
                    if not line:
                        break
                    output_lines.append(line.strip())
                except Exception:
                    break
            
            # Look for tunnel URL in the output
            for line in output_lines:
                if "tunneled with tls termination" in line:
                    import re
                    url_match = re.search(r'https?://[^\s]+\.(localhost\.run|lhr\.life)', line)
                    if url_match:
                        public_url = url_match.group(0)
                        save_tunnel_info(public_url, port)
                        break
        except Exception:
            pass  # Silently handle any errors
        
        # Keep process running silently
        tunnel_process.wait()
        
    except Exception:
        pass  # Silently handle tunnel creation errors

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
        # Start the tunnel in a separate thread (completely silent)
        tunnel_thread = threading.Thread(target=create_public_tunnel, args=(target_port,))
        tunnel_thread.daemon = True
        tunnel_thread.start()
        
        # Give the tunnel a moment to start
        time.sleep(1)
    
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
        # Run uvicorn with minimal logging
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=target_port,
            log_level="info",  # Reduce log verbosity
            access_log=True    # Keep access logs but clean
        )
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    finally:
        cleanup_tunnel()


