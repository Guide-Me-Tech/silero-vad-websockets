import http.server
import socketserver
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
PORT = 8090
DIRECTORY = "."  # Current directory

class Handler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))

def main():
    """Start the HTTP server."""
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        logger.info(f"HTTP server started on port {PORT}")
        logger.info(f"Open http://localhost:{PORT}/home.html in your browser")
        httpd.serve_forever()

if __name__ == "__main__":
    main() 