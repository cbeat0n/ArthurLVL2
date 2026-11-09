### scheduler.py is a modular and agnostic framework that runs one function per horizon on a
### background thread, aligned to the horizon boundary, so trading does not depend on a browser tab.

## External Dependencies
import threading # Library for background threads
import time # Library for time
import traceback # Library for printing errors from the thread
from datetime import datetime , timezone # Library for time objects


class CycleScheduler:
    """
    CycleScheduler calls run_cycle once per period on its own thread. Cycles fire at multiples of
    the period since UTC midnight (plus a small offset, so the exchange has published the last
    candle), never overlap, and keep going after an error.

    Functions:
    __init__ : initialization function
    seconds_until_next : seconds to the next cycle
    start : starts the thread
    stop : stops the thread
    status : text for the dashboard
    """

    def __init__(self , period_s : float , run_cycle , offset_s : float = 2.0 , name : str = "arthur-cycle"):
        """
        __init__ initializes a CycleScheduler. It returns nothing.

        Inputs:
        self : self
        period_s : float - seconds per cycle, the horizon in seconds
        run_cycle : callable - called once per cycle. Return False to stop the scheduler,
        anything else to continue. Exceptions are caught, recorded, and the loop continues.
        offset_s : float - seconds after each boundary before the cycle runs
        name : str - thread name
        """

        self.period_s = float(period_s)
        self.offset_s = float(offset_s)
        self.run_cycle = run_cycle
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._loop , name = name , daemon = True)
        self._lock = threading.Lock()   # guards the status fields below

        self.cycles = 0 # cycles run so far
        self.failures = 0 # consecutive failed cycles
        self.last_started = None # epoch seconds
        self.last_duration = None # seconds
        self.last_error = None # text of the last error, None if the last cycle was fine
        self.next_run = None # epoch seconds
        self.overruns = 0 # cycles that took longer than the period

    def seconds_until_next(self , now : float = None) -> float:
        """
        seconds_until_next gives the seconds until the next boundary plus offset. Boundaries are
        multiples of the period since UTC midnight, and the day always restarts at midnight.
        It returns a float.

        Inputs:
        self : self
        now : float - epoch seconds, defaults to the current time
        """

        now = time.time() if now is None else now
        midnight = now - (now % 86400)
        k = int((now - midnight - self.offset_s) // self.period_s) + 1
        nxt = midnight + k * self.period_s + self.offset_s
        nxt = min(nxt , midnight + 86400 + self.offset_s) # a new day restarts the grid
        return nxt - now

    def start(self):
        """
        start starts the background thread once. It returns nothing.
        """

        if not self._thread.is_alive() and not self._stop.is_set():
            self._thread.start()

    def stop(self , timeout : float = 5.0):
        """
        stop asks the loop to end and waits for it, unless called from the loop itself.
        It returns nothing.

        Inputs:
        timeout : float - seconds to wait for the thread
        """

        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def _loop(self):
        while True:
            wait = self.seconds_until_next()
            with self._lock:
                self.next_run = time.time() + wait
            if self._stop.wait(wait): # woken early only by stop()
                return

            started = time.time()
            error = None
            result = None
            try:
                result = self.run_cycle()
            except Exception as e: # one bad cycle must not end the bot
                error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            duration = time.time() - started

            with self._lock:
                self.cycles += 1
                self.last_started = started
                self.last_duration = duration
                self.last_error = error
                self.failures = self.failures + 1 if error else 0
                if duration > self.period_s:
                    self.overruns += 1 # missed boundaries are skipped, never queued

            if result is False:
                return

    def status(self) -> dict:
        """
        status gives text for the dashboard cards. It returns a dict.
        """

        fmt = lambda t: "-" if t is None else datetime.fromtimestamp(t , timezone.utc).strftime("%H:%M:%S")
        with self._lock:
            return {
                "Scheduler" : "running" if self._thread.is_alive() else "stopped",
                "Last cycle (UTC)" : fmt(self.last_started),
                "Cycle seconds" : "-" if self.last_duration is None else f"{self.last_duration:.1f}",
                "Next cycle (UTC)" : fmt(self.next_run),
                "Cycles run" : str(self.cycles),
                "Failed in a row" : str(self.failures),
                "Last error" : self.last_error or "none",
            }
