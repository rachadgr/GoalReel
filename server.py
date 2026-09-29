from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/health': self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(b'{"status":"ok","service":"goalreel"}')
  else: self.send_error(404)
if __name__=='__main__': HTTPServer(('0.0.0.0',8000),H).serve_forever()
