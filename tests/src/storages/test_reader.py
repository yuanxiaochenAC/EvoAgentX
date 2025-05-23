import unittest

from evoagentx.rag.readers import LLamaIndexReader


PATH = r"D:\Docker_store\store\MyKits\Project\EvoAgentX\debug"
class TestReader(unittest.TestCase):

    def setUp(self):
        self.reader = LLamaIndexReader(recursive=True, num_files_limits=10, 
                                       num_workers=2)
    
    def test_doc(self,):
        print("====================Test Docx loadding====================")
        doc = self.reader.load(PATH, filter_file_by_suffix=[".docx"])
        print(doc)
        print(f"Doc content:\n {doc[0].text}")

    def test_pdf(self,):
        print("====================Test PDF loadding====================")
        doc = self.reader.load(PATH, filter_file_by_suffix=[".pdf"])
        print(doc)
        print(f"Doc content:\n {doc[0].text}")


if __name__ == "__main__":
    test = TestReader()
    test.setUp()
    test.test_doc()
    test.test_pdf()