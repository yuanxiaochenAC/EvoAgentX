import unittest

from evoagentx.rag import SimpleChunker, HierarchicalChunker, SemanticChunker
from evoagentx.rag.readers import LLamaIndexReader


PATH = r"/data2/caizijian/shibie/code/EvoAgentX/debug/doc"
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
        import pdb;pdb.set_trace()
        print(doc)
        print(f"Doc content:\n {doc[0].text}")
        doc_corps1 = self.chunker1.chunk(doc)
        # doc_corps2 = self.chunker2.chunk(doc)
        doc_corps3 = self.chunker3.chunk(doc)


if __name__ == "__main__":
    test = TestChunker()
    test.setUp()
    test.test_doc()