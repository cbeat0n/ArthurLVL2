### http_helpers.py is a modular and agnostic framework that gives every vendor
### request a timeout, safe retries and clear errors.

## External Dependencies
import requests # Library for HTTP requests
from requests.adapters import HTTPAdapter # Library for connection pooling and retry
from urllib3.util.retry import Retry # Library for retry policy


class OandaError(ValueError):
    """
    OandaError is raised when OANDA returns a bad status or a body that is not JSON.
    It subclasses ValueError so existing `except ValueError` handlers keep working.
    """


class OandaSession(requests.Session):
    """
    OandaSession is a requests.Session with a default timeout on every call and automatic
    retries. GET requests retry on 429 / 5xx and on connection or read errors. Order
    requests (POST / PUT) are never retried after they were sent, so a slow response
    can't place the same order twice. Connection failures before sending are retried.
    """

    def __init__(self , timeout : tuple = (3.05 , 10) , retries : int = 3):
        super().__init__()
        self.default_timeout = timeout

        policy = Retry(total = retries , connect = retries , read = retries , status = retries ,
                       backoff_factor = 0.5 ,
                       status_forcelist = (429 , 500 , 502 , 503 , 504) ,
                       allowed_methods = frozenset(["GET"]) ,
                       respect_retry_after_header = True ,
                       raise_on_status = False)
        adapter = HTTPAdapter(max_retries = policy)
        self.mount("https://" , adapter)
        self.mount("http://" , adapter)

    def request(self , method , url , **kwargs):
        kwargs.setdefault("timeout" , self.default_timeout)
        return super().request(method , url , **kwargs)


def json_or_raise(response : requests.Response , what : str = "OANDA request"):
    """
    json_or_raise returns the JSON body of a 200 response, and raises OandaError for any other
    status or for a body that is not JSON (e.g a gateway or waiting room HTML page).
    """

    if response.status_code != 200:
        raise OandaError(f"{what} failed with {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError:
        raise OandaError(f"{what} returned a non-JSON body: {response.text[:200]}")