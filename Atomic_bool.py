import threading

class AtomicBoolean:
    def __init__(self, initial=True):
        self.value = initial
        self._lock = threading.Lock()

    def false(self):
        with self._lock:
            self.value = False
            return self.value

    def true(self):
        with self._lock:
            self.value = True
            return self.value

    def get(self):
        with self._lock:
            return self.value
