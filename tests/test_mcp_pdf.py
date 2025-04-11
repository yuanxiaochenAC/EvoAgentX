import asyncio
import os
import sys
import json
import tempfile
import shutil
import http.server
import socketserver
import threading
import time
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.insert(0, str(Path(__file__).parent.parent))

from evoagentx.tools.mcp import MCPClient, MCPToolkit

# Sample config for PDF Reader MCP server
config = {
  "mcpServers": {
    "pdf-reader-mcp": {
      "command": "npx",
      "args": ["@sylphlab/pdf-reader-mcp"],
      "name": "PDF Reader (npx)"
    }
  }
}

class SimpleHTTPServerThread(threading.Thread):
    """Simple HTTP server for serving PDF files"""
    
    def __init__(self, directory, port=8000):
        super().__init__(daemon=True)
        self.directory = directory
        self.port = port
        self.httpd = None
        
    def run(self):
        """Run the HTTP server in a separate thread"""
        handler = http.server.SimpleHTTPRequestHandler
        
        # Remember current directory
        original_dir = os.getcwd()
        
        try:
            # Change to the directory containing the files to serve
            os.chdir(self.directory)
            
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                print(f"Serving files from {self.directory} at port {self.port}")
                self.httpd = httpd
                httpd.serve_forever()
        finally:
            # Restore original directory
            os.chdir(original_dir)
            
    def stop(self):
        """Stop the HTTP server"""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            print("HTTP server stopped")

class PDFProcessor:
    """Class to process PDFs using MCP toolkit"""
    
    def __init__(self, pdf_client):
        """Initialize with an MCP client"""
        self.pdf_client = pdf_client
        
    async def process_pdf_url(self, pdf_url, page_numbers=None, include_full_text=True, include_metadata=True, include_page_count=True):
        """Process a PDF document from a URL using MCP toolkit
        
        Args:
            pdf_url: URL to the PDF file
            page_numbers: Optional specific page numbers to process (array of integers)
            include_full_text: Whether to include the full text content
            include_metadata: Whether to include metadata and info objects
            include_page_count: Whether to include the total page count
            
        Returns:
            Formatted result from the operation
        """
        # Create the MCP toolkit
        toolkit = MCPToolkit(servers=[self.pdf_client])
        
        try:
            # Connect to the PDF server
            await toolkit.connect()
            
            if not toolkit.is_connected():
                return "Failed to connect to PDF server"
            
            # Use the correct parameter structure for read_pdf tool with URL
            source = {"url": pdf_url}
            if page_numbers:
                source["pages"] = page_numbers
                
            args = {
                "sources": [source],
                "include_full_text": include_full_text,
                "include_metadata": include_metadata,
                "include_page_count": include_page_count
            }
                
            # Call the read_pdf tool
            result = await self._read_pdf(toolkit, args)
            return result
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
        finally:
            # Ensure we disconnect
            if toolkit.is_connected():
                await toolkit.disconnect()
    
    async def process_pdf(self, pdf_path, page_numbers=None, include_full_text=True, include_metadata=True, include_page_count=True):
        """Process a PDF document using MCP toolkit (path-based, kept for compatibility)
        
        Args:
            pdf_path: Path to the PDF file
            page_numbers: Optional specific page numbers to process (array of integers)
            include_full_text: Whether to include the full text content
            include_metadata: Whether to include metadata and info objects
            include_page_count: Whether to include the total page count
            
        Returns:
            Formatted result from the operation
        """
        # Create the MCP toolkit
        toolkit = MCPToolkit(servers=[self.pdf_client])
        
        try:
            # Connect to the PDF server
            await toolkit.connect()
            
            if not toolkit.is_connected():
                return "Failed to connect to PDF server"
                
            # Convert to relative path - use the file name only
            # This assumes the PDF server runs in the project root
            rel_path = os.path.basename(pdf_path)
            
            # Use the correct parameter structure for read_pdf tool
            source = {"path": rel_path}
            if page_numbers:
                source["pages"] = page_numbers
                
            args = {
                "sources": [source],
                "include_full_text": include_full_text,
                "include_metadata": include_metadata,
                "include_page_count": include_page_count
            }
                
            # Call the read_pdf tool
            result = await self._read_pdf(toolkit, args)
            return result
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
        finally:
            # Ensure we disconnect
            if toolkit.is_connected():
                await toolkit.disconnect()
    
    async def _read_pdf(self, toolkit, args):
        """Call the read_pdf tool with the correct arguments"""
        for server in toolkit.servers:
            tools_info = server.tools_info()
            if any(tool["name"] == "read_pdf" for tool in tools_info):
                result = await server.call_tool("read_pdf", args)
                return self._format_result(result)
                
        return "Tool 'read_pdf' not found"
    
    def _format_result(self, result):
        """Format the result from the MCP tool"""
        if not result:
            return "No result returned"
            
        if hasattr(result, 'text'):
            return result.text
            
        if isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'text'):
            return result[0].text
            
        # Try to pretty format if it looks like JSON
        if isinstance(result, dict):
            try:
                return json.dumps(result, indent=2)
            except:
                pass
                
        return str(result)


async def pdf_demo():
    """Demo that demonstrates using the PDF MCP server"""
    print("\n==== PDF MCP DEMO ====")
    
    # Get the path to the test PDF
    # Store it in tests directory for easy relative path access
    pdf_path = os.path.join(os.path.dirname(__file__), "test_pdf.pdf")
    
    # Get just the filename for display
    pdf_filename = os.path.basename(pdf_path)
    
    if not os.path.exists(pdf_path):
        print(f"Error: Test PDF not found at {pdf_path}")
        print(f"Please create a PDF file named 'test_pdf.pdf' in the tests directory before running this demo.")
        return
    
    print(f"Using test PDF at: {pdf_path}")
    print(f"Using relative file path: {pdf_filename}")
    
    # Make sure to copy the PDF to the current working directory if needed
    current_dir = os.getcwd()
    if os.path.dirname(pdf_path) != current_dir:
        dest_path = os.path.join(current_dir, pdf_filename)
        print(f"Copying PDF to current working directory: {current_dir}")
        shutil.copy2(pdf_path, dest_path)
        print(f"PDF copied to: {dest_path}")
    
    # Start a simple HTTP server to serve the PDF file
    http_port = 8000
    server_thread = SimpleHTTPServerThread(current_dir, http_port)
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(1)
    
    # Construct the URL to the PDF
    pdf_url = f"http://localhost:{http_port}/{pdf_filename}"
    print(f"Serving PDF at URL: {pdf_url}")
    
    try:
        # Create the PDF client
        pdf_client = MCPClient(
            command_or_url=config["mcpServers"]["pdf-reader-mcp"]["command"],
            args=config["mcpServers"]["pdf-reader-mcp"]["args"],
            env=os.environ.copy()
        )
        
        # Connect to the server first to list available tools
        toolkit = MCPToolkit(servers=[pdf_client])
        try:
            print("Connecting to PDF server to list available tools...")
            await toolkit.connect()
            
            if not toolkit.is_connected():
                print("Failed to connect to PDF server")
                return
                
            # Get detailed information about available tools
            detailed_info = toolkit.detailed_tools_info()
            
            print("\n=== Available PDF Tools ===")
            for server_name, server_tools in detailed_info.items():
                print(f"\nServer: {server_name}")
                print(f"  Number of tools: {len(server_tools)}")
                
                for i, tool in enumerate(server_tools):
                    print(f"\n  TOOL {i+1}: {tool['name']}")
                    print(f"    Description: {tool['description']}")
                    print(f"    Required parameters: {tool['required_params']}")
                    print("    Parameters:")
                    for param_name, param_schema in tool['parameters'].items():
                        param_type = param_schema.get('type', 'any')
                        param_desc = param_schema.get('description', 'No description')
                        required = "Required" if param_name in tool['required_params'] else "Optional"
                        print(f"      - {param_name} ({param_type}, {required}): {param_desc}")
        finally:
            if toolkit.is_connected():
                await toolkit.disconnect()
        
        # Create a PDF processor
        pdf_processor = PDFProcessor(pdf_client)
        
        # Test different parameter combinations
        test_cases = [
            {
                "name": "Full Document (URL)",
                "params": {
                    "include_full_text": True,
                    "include_metadata": True, 
                    "include_page_count": True
                }
            },
            {
                "name": "Metadata Only (URL)",
                "params": {
                    "include_full_text": False,
                    "include_metadata": True,
                    "include_page_count": True
                }
            },
            {
                "name": "Specific Pages (URL)",
                "params": {
                    "page_numbers": [1], 
                    "include_full_text": True,
                    "include_metadata": False
                }
            }
        ]
        
        # Test with URL-based access
        print("\n=== TESTING URL-BASED ACCESS ===")
        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['name']} ---")
            try:
                # Process the PDF with the specified parameters
                result = await pdf_processor.process_pdf_url(
                    pdf_url=pdf_url,
                    **test_case['params']
                )
                
                # Print result (truncate if too long)
                if len(str(result)) > 1000:
                    print(f"Result: {str(result)[:1000]}...\n(truncated)")
                else:
                    print(f"Result: {result}")
                    
            except Exception as e:
                print(f"Error executing test case: {e}")
        
        # For comparison, also try the path-based approach
        print("\n=== TESTING PATH-BASED ACCESS (May Fail) ===")
        try:
            result = await pdf_processor.process_pdf(
                pdf_path=pdf_filename,
                include_full_text=True,
                include_metadata=True
            )
            print("Result:", result)
        except Exception as e:
            print(f"Error with path-based access: {e}")
    
    finally:
        # Stop the HTTP server
        server_thread.stop()
        print("HTTP server thread stopped")


async def run_demo():
    """Run the PDF MCP demo"""
    try:
        await pdf_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_demo())