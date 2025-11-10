from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any
import time

class TimeCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        self.start_time = time.time()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(f"\n--- Tempo de execução: {time.time() - self.start_time:.2f} segundos ---\n")