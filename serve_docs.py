"""Preview the hosted (GitHub Pages) version of the dashboard locally: python serve_docs.py
Open http://[::1]:8800/ (not localhost, so the page behaves exactly like the hosted one).
Uses HTTP/1.1 because the default HTTP/1.0 server cuts off big files on some Windows setups."""
import functools
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass


class Server(ThreadingHTTPServer):
    address_family = __import__("socket").AF_INET6


if __name__ == "__main__":
    docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    Server(("::", 8800), functools.partial(Handler, directory=docs)).serve_forever()
