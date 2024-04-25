''This module contain locks that will be shared among multiple modules'''

import threading

# Create a lock to prevent multiple threads from accessing the buffer at the same time
lock = threading.Lock()
