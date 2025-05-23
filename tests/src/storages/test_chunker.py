import unittest

from evoagentx.rag import SimpleChunker
from evoagentx.rag.readers import LLamaIndexReader


PATH = r"D:\Docker_store\store\MyKits\Project\EvoAgentX\debug"
class TestReader(unittest.TestCase):

    def setUp(self):
        self.reader = LLamaIndexReader(recursive=True, num_files_limits=10, 
                                       num_workers=2)
        self.chunker = SimpleChunker()
    
    def test_doc(self,):
        print("====================Test Docx loadding====================")
        doc = self.reader.load(PATH, filter_file_by_suffix=[".docx"])
        print(doc)
        print(f"Doc content:\n {doc[0].text}")
        import pdb;pdb.set_trace()
        doc_corps = self.chunker.chunk(doc)


if __name__ == "__main__":
    test = TestReader()
    test.setUp()
    test.test_doc()