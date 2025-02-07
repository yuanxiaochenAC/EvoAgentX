import os 
import unittest
from pydantic import Field, field_validator
from typing import Optional, Union, List, Dict
from evoagentx.core.module import BaseModule

class ToyTool:

    def __init__(self, name: str, key: str, **kwargs):
        self.name = name
        self.key = key
        self.kwargs = kwargs

def get_tool(name, key):
    return ToyTool(name, key)

class Test(BaseModule):

    k1: Union[str, int] 
    k3: list = Field(default=[1,2])

class TestSubClass(Test):
    pass 

class Test2(BaseModule):

    k4: Optional[str] = Field(default=None, description="name")
    k5: str = Field(description="key")
    k6: Optional[ToyTool] = Field(default=None)

    @field_validator("k4")
    @classmethod
    def validate_k4(cls, value):
        if value == "k4_value":
            raise NotImplementedError("the method for \"k4=k4_value\" is not implemented!")
        return value

    def init_module(self):
        if self.k6 is None:
            if self.k4 is not None and self.k5 is not None:
                self.k6 = ToyTool(self.k4, self.k5)
            else:
                raise ValueError(f"either k4 and k5 is None!")

class Test2SubClass(Test2):

    test2_subclass_variable: int = Field(default=0)

class Test3(BaseModule):

    k7: Test 
    k8: int 
    k9: Test2 

class Test3SubClass(Test3):
    pass 

class Test4(BaseModule):

    k10: List[Test] = None 
    k11: List[Test2] = None
    k12: Dict[str, Test3] = None
    k13: Dict[str, int] = None 

class TestModule(unittest.TestCase):

    def setUp(self):
        self.save_file = "tests/core/saved_module.json"

    def test_initialization(self):

        module1 = Test(k1=100)
        self.assertEqual(module1.k1, 100)
        self.assertEqual(module1.k3, [1, 2])
        module12 = Test(k1=100, k3=[200, 300])
        self.assertEqual(module12.k3, [200, 300])

        module2 = Test2(k4="k4_value_valid", k5="k5_value")
        self.assertEqual(module2.k4, "k4_value_valid")
        self.assertEqual(module2.k5, "k5_value")
        self.assertEqual(module2.k6.name, "k4_value_valid")
        self.assertEqual(module2.k6.key, "k5_value")

        module3 = Test3(k7=module1, k8=10, k9=module2)
        self.assertEqual(module3.k7, module1)
        self.assertEqual(module3.k9, module2)
    
    def test_from_dict(self):

        module = Test3.from_dict(
            {
                "k7": {
                    "k1": "k1_value", 
                    "k3": [100, 200], 
                }, 
                "k8": 10, 
                "k9": Test2(k4="k4_value_valid", k5="k5_value"),
            }
        )
        self.assertEqual(module.k7.k1, "k1_value")
        self.assertEqual(module.k7.k3, [100, 200])
        self.assertEqual(module.k8, 10)
        self.assertEqual(module.k9.k6.name, "k4_value_valid")
        self.assertEqual(module.k9.k6.key, "k5_value") 
    
    def test_from_json(self):

        json_data = """
        {
            "k7": {
                "k1": "k1_value", 
                "k3": [100, 200], 
            },
            "k8": 10, 
            "k9": {
                "k4": "k4_value_valid", 
                "k5": "k5_value", 
            }
        }
        """
        module = Test3.from_json(json_data)
        self.assertEqual(module.k7.k1, "k1_value")
        self.assertEqual(module.k7.k3, [100, 200])
        self.assertEqual(module.k8, 10)
        self.assertEqual(module.k9.k6.name, "k4_value_valid")
        self.assertEqual(module.k9.k6.key, "k5_value") 
    
    def test_from_str(self):

        str_data = """
        there might be some text before the json data. 

        an irrelevant json data:
        {
            "k1": "k1",
            "k3": 11, 
        }

        true json data: 
        {
            "k7": {
                "k1": "k1_value", 
                "k3": [100, 200], 
            },
            "k8": 10, 
            "k9": {
                "k4": "k4_value_valid", 
                "k5": "k5_value", 
            }
        }
        
        some text after the json data. 
        """
        module = Test3.from_str(str_data)
        self.assertEqual(module.k7.k1, "k1_value")
        self.assertEqual(module.k7.k3, [100, 200])
        self.assertEqual(module.k8, 10)
        self.assertEqual(module.k9.k6.name, "k4_value_valid")
        self.assertEqual(module.k9.k6.key, "k5_value") 
    
    def test_save_module(self):

        module1 = Test(k1="k1_value", k3=[100, 200])
        module2 = Test2(k4="k4_value_valid", k5="k5_value")
        module3 = Test3(k7=module1, k8=10, k9=module2)
        module3.save_module(self.save_file, use_indent=True)
        self.assertTrue(os.path.exists(self.save_file))

        module = Test3.from_file(self.save_file)
        self.assertEqual(module.k7.k1, "k1_value")
        self.assertEqual(module.k7.k3, [100, 200])
        self.assertEqual(module.k8, 10)
        self.assertEqual(module.k9.k6.name, "k4_value_valid")
        self.assertEqual(module.k9.k6.key, "k5_value")

    def test_subclass(self):

        d1 = {
            "k10": [{"k1": "k1_value"}], 
            "k11": [{"k4": "k4_valid_value1", "k5": "k5_value1"}, {"k4": "k4_valid_value2", "k5": "k5_value2"}],
            "k12": {
                "key": {
                    "k7": {"k1": "k1_value2"}, 
                    "k8": 11, 
                    "k9": {"k4": "k4_valid_value3", "k5": "k5_value3"}
                }
            }
        }
        module = Test4.from_dict(d1)
        self.assertTrue(isinstance(module.k10[0], Test) and module.k10[0].class_name=="Test")
        self.assertTrue(isinstance(module.k11[0], Test2) and isinstance(module.k11[1], Test2) \
                        and module.k11[0].class_name=="Test2" and module.k11[1].class_name=="Test2")
        self.assertTrue(isinstance(module.k12["key"], Test3) and module.k12["key"].class_name=="Test3")
        self.assertTrue(isinstance(module.k12["key"].k7, Test) and module.k12["key"].k7.class_name=="Test")
        self.assertTrue(isinstance(module.k12["key"].k9, Test2) and module.k12["key"].k9.class_name=="Test2")

        d2 = {
            "k10": [{"k1": "k1_value"}], 
            "k11": [{"class_name": "Test2SubClass", "k4": "k4_valid_value1", "k5": "k5_value1"}, {"k4": "k4_valid_value2", "k5": "k5_value2"}],
            "k12": {
                "key": {
                    "k7": {"class_name": "TestSubClass", "k1": "k1_value2"}, 
                    "k8": 11, 
                    "k9": {"k4": "k4_valid_value3", "k5": "k5_value3"}
                }
            }, 
            "k13": {
                "key2": 0
            }
        }
        module = Test4.from_dict(d2)
        self.assertTrue(isinstance(module.k10[0], Test) and module.k10[0].class_name=="Test")
        self.assertTrue(isinstance(module.k11[0], Test2SubClass) and isinstance(module.k11[1], Test2) \
                        and module.k11[0].class_name=="Test2SubClass" and module.k11[1].class_name=="Test2")
        self.assertEqual(module.k11[0].test2_subclass_variable, 0)
        self.assertTrue(isinstance(module.k12["key"], Test3) and module.k12["key"].class_name=="Test3")
        self.assertTrue(isinstance(module.k12["key"].k7, TestSubClass) and module.k12["key"].k7.class_name=="TestSubClass")
        self.assertTrue(isinstance(module.k12["key"].k9, Test2) and module.k12["key"].k9.class_name=="Test2")
        self.assertTrue(isinstance(module.k13, dict))
    
    def test_subclass_from_init(self):

        test2_instance = Test2(k4="k4_valid_value1", k5="k5_value1")
        test4_instance = Test4(
            k10 = [{"k1": "k1_value"}], 
            k11 = [test2_instance, {"k4": "k4_valid_value2", "k5": "k5_valid_value2"}, {"class_name": "Test2SubClass", "k4": "k4_valid_value3", "k5": "k5_value3", "test2_subclass_variable": 888}], 
            k12 = {
                "key": {
                    "class_name": "Test3SubClass", 
                    "k7": {"class_name": "TestSubClass", "k1": "k1_value2"}, 
                    "k8": 11, 
                    "k9": {"k4": "k4_valid_value4", "k5": "k5_value4"}
                }
            }, 
            k13 = {
                "key2": 999
            }
        )
        self.assertEqual(test4_instance.k10[0].k1, "k1_value")
        self.assertTrue(isinstance(test4_instance.k11[0], Test2))
        self.assertTrue(isinstance(test4_instance.k11[2], Test2SubClass))
        self.assertEqual(test4_instance.k11[2].test2_subclass_variable, 888)
        self.assertTrue(isinstance(test4_instance.k12["key"], Test3SubClass))
        self.assertTrue(isinstance(test4_instance.k12["key"].k7, TestSubClass))

    def tearDown(self):
        if os.path.exists(self.save_file):
            os.remove(self.save_file)

if __name__ == "__main__":
    unittest.main()

