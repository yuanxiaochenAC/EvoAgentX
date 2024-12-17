import threading
from contextlib import contextmanager

class Callback:

    """
    a base class for callbacks 
    """

    def on_error(self, exception, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        try:
            result = self.run(*args, **kwargs)
        except Exception as e:
            self.on_error(e, *args, kwargs)
            raise e 
        return result
    
    def run(self, *args, **kwargs):
        raise NotImplementedError(f"run is not implemented for {type(self).__name__}!")


class CallbackManager:

    def __init__(self):
        self.local_data = threading.local()
        self.local_data.callbacks = {}

    def set_callback(self, callback_type: str, callback: Callback):
        self.local_data.callbacks[callback_type] = callback

    def get_callback(self, callback_type: str):
        return self.local_data.callbacks.get(callback_type, None)

    def clear_callback(self, callback_type: str):
        if callback_type in self.local_data.callbacks:
            del self.local_data.callbacks[callback_type]

    def clear_all(self):
        self.local_data.callbacks.clear()

callback_manager = CallbackManager()


class DeferredExceptionHandler(Callback):

    def __init__(self):
        self.exceptions = [] 
    
    def add(self, exception):
        self.exceptions.append(exception)
    

@contextmanager
def exception_buffer():
    exception_handler = DeferredExceptionHandler()
    callback_manager.set_callback("exception_buffer", exception_handler)
    try:
        yield exception_handler
    finally:
        callback_manager.clear_callback("exception_buffer")
    

