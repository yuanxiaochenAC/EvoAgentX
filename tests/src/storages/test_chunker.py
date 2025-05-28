import time
import unittest

from evoagentx.rag import SimpleChunker, HierarchicalChunker, SemanticChunker
from evoagentx.rag.readers import LLamaIndexReader


class Timer:
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0
    
    def __enter__(self, *args, **kwargs):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        print(f"This operator cost time: {round(self.end_time - self.start_time, 4)}s")


PATH = r"debug/doc"
class TestChunker(unittest.TestCase):

    def setUp(self):
        self.reader = LLamaIndexReader(recursive=True, num_files_limits=10, 
                                       num_workers=2)
        self.chunker1 = SimpleChunker()
        # self.chunker2 = SemanticChunker()
        self.chunker3 = HierarchicalChunker()
    
    def test_doc(self,):
        print("====================Test Docx loadding====================")
        doc = self.reader.load(PATH, filter_file_by_suffix=[".docx"])
        print(doc)
        print(f"Doc content:\n {doc[0].text}")
        with Timer():
            doc_corps1 = self.chunker1.chunk(doc)
        # doc_corps2 = self.chunker2.chunk(doc)
        doc_corps3 = self.chunker3.chunk(doc)


if __name__ == "__main__":
    test = TestChunker()
    test.setUp()
    test.test_doc()