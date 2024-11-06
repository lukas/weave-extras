from weave import Model
import weave
import weaveindex
from llama_index.core.query_engine import BaseQueryEngine


class QAModel(Model):
    query_engine: BaseQueryEngine
    model_name: str

    @weave.op()
    def predict(self, question: str) -> dict:
        # Model logic goes here
        prediction = weaveindex.query(question, self.query_engine,
                                      self.model_name)
        return {'pred': prediction}
