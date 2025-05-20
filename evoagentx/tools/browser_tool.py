from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from typing import Dict, Any, List, Callable, Optional
from pydantic import Field
from .tool import Tool
from evoagentx.core.logging import logger

class BrowserTool(Tool):
    """
    A tool for interacting with web browsers using Selenium.
    Allows agents to navigate to URLs, interact with elements, extract information,
    and more from web pages.
    """
    
    timeout: int = Field(default=10, description="Default timeout in seconds for browser operations")
    
    def __init__(
        self,
        name: str = "Browser Tool",
        browser_type: str = "chrome",
        headless: bool = False,
        timeout: int = 10,
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):
        """
        Initialize the browser tool with Selenium WebDriver.
        
        Args:
            name (str): Name of the tool
            browser_type (str): Type of browser to use ('chrome', 'firefox', 'safari', 'edge')
            headless (bool): Whether to run the browser in headless mode
            timeout (int): Default timeout in seconds for browser operations
            schemas (List[dict], optional): Tool schemas
            descriptions (List[str], optional): Tool descriptions
            tools (List[Callable], optional): Tool functions
            **kwargs: Additional keyword arguments for parent class initialization
        """
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
        
        # Pass to parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            **kwargs
        )
        
        self.timeout = timeout
        self.browser_type = browser_type
        self.headless = headless
        self.driver = None
        
    def initialize_browser(self) -> Dict[str, Any]:
        """
        Initialize and start the browser session.
        
        Returns:
            Dict[str, Any]: Status information about the browser initialization
        """
        try:
            if self.driver:
                # Close any existing session
                try:
                    self.driver.quit()
                except Exception as e:
                    logger.warning(f"Error closing existing browser session: {str(e)}")
                    
            options = None
            if self.browser_type == "chrome":
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
                # Create service with Chrome executable path
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            elif self.browser_type == "firefox":
                from selenium.webdriver.firefox.options import Options
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Firefox(options=options)
            elif self.browser_type == "safari":
                self.driver = webdriver.Safari()
            elif self.browser_type == "edge":
                from selenium.webdriver.edge.options import Options
                options = Options()
                if self.headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Edge(options=options)
            else:
                return {"status": "error", "message": f"Unsupported browser type: {self.browser_type}"}
            
            self.driver.set_page_load_timeout(self.timeout)
            return {"status": "success", "message": f"Browser {self.browser_type} initialized successfully"}
        except Exception as e:
            logger.error(f"Error initializing browser: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def navigate_to_url(self, url: str, timeout: int = None) -> Dict[str, Any]:
        """
        Navigate the browser to a specific URL.
        
        Args:
            url (str): The URL to navigate to
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the navigation result
        """
        if not self.driver:
            init_result = self.initialize_browser()
            if init_result["status"] == "error":
                return init_result
                
        timeout = timeout or self.timeout
        try:
            self.driver.get(url)
            return {
                "status": "success", 
                "url": url,
                "title": self.driver.title,
                "current_url": self.driver.current_url
            }
        except TimeoutException:
            return {"status": "timeout", "message": f"Timed out loading URL: {url}"}
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def find_element(self, selector: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Find an element on the current page and return information about it.
        
        Args:
            selector (str): The selector to find the element
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the found element
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        timeout = timeout or self.timeout
        selector_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }
        
        try:
            by_type = selector_map.get(selector_type.lower())
            if not by_type:
                return {"status": "error", "message": f"Invalid selector type: {selector_type}"}
                
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            
            # Extract element properties
            element_properties = {
                "text": element.text,
                "tag_name": element.tag_name,
                "is_displayed": element.is_displayed(),
                "is_enabled": element.is_enabled(),
            }
            
            # Get attributes
            try:
                element_properties["href"] = element.get_attribute("href")
            except StaleElementReferenceException:
                logger.warning(f"Element became stale when trying to get href attribute for {selector}")
            except Exception as e:
                logger.warning(f"Could not get href attribute for {selector}: {str(e)}")
                
            try:
                element_properties["id"] = element.get_attribute("id")
            except StaleElementReferenceException:
                logger.warning(f"Element became stale when trying to get id attribute for {selector}")
            except Exception as e:
                logger.warning(f"Could not get id attribute for {selector}: {str(e)}")
                
            try:
                element_properties["class"] = element.get_attribute("class")
            except StaleElementReferenceException:
                logger.warning(f"Element became stale when trying to get class attribute for {selector}")
            except Exception as e:
                logger.warning(f"Could not get class attribute for {selector}: {str(e)}")
                
            return {
                "status": "success",
                "element": element_properties
            }
        except TimeoutException:
            return {"status": "not_found", "message": f"Element not found with {selector_type}: {selector}"}
        except Exception as e:
            logger.error(f"Error finding element {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def find_multiple_elements(self, selector: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Find multiple elements on the current page and return information about them.
        
        Args:
            selector (str): The selector to find the elements
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Information about the found elements
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        timeout = timeout or self.timeout
        selector_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }
        
        try:
            by_type = selector_map.get(selector_type.lower())
            if not by_type:
                return {"status": "error", "message": f"Invalid selector type: {selector_type}"}
                
            # First check if at least one element exists
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            
            # Then get all matching elements
            elements = self.driver.find_elements(by_type, selector)
            
            # Extract element properties
            elements_properties = []
            for idx, element in enumerate(elements):
                try:
                    element_properties = {
                        "index": idx,
                        "text": element.text,
                        "tag_name": element.tag_name,
                        "is_displayed": element.is_displayed(),
                        "is_enabled": element.is_enabled(),
                    }
                    
                    # Get attributes
                    try:
                        element_properties["href"] = element.get_attribute("href")
                    except (StaleElementReferenceException, Exception):
                        pass
                        
                    try:
                        element_properties["id"] = element.get_attribute("id")
                    except (StaleElementReferenceException, Exception):
                        pass
                        
                    try:
                        element_properties["class"] = element.get_attribute("class")
                    except (StaleElementReferenceException, Exception):
                        pass
                        
                    elements_properties.append(element_properties)
                except StaleElementReferenceException:
                    logger.warning(f"Element {idx} became stale while extracting properties")
                except Exception as e:
                    logger.warning(f"Error extracting properties for element {idx}: {str(e)}")
                
            return {
                "status": "success",
                "count": len(elements_properties),
                "elements": elements_properties
            }
        except TimeoutException:
            return {"status": "not_found", "message": f"No elements found with {selector_type}: {selector}"}
        except Exception as e:
            logger.error(f"Error finding elements {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def wait_for_element(self, selector: str, condition: str = "presence", selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Wait for an element to meet a specific condition.
        
        Args:
            selector (str): The selector to find the element
            condition (str): The condition to wait for ('presence', 'visibility', 'clickable', 'invisibility')
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Result of the wait operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        timeout = timeout or self.timeout
        selector_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }
        
        condition_map = {
            "presence": EC.presence_of_element_located,
            "visibility": EC.visibility_of_element_located,
            "clickable": EC.element_to_be_clickable,
            "invisibility": EC.invisibility_of_element_located,
        }
        
        try:
            by_type = selector_map.get(selector_type.lower())
            if not by_type:
                return {"status": "error", "message": f"Invalid selector type: {selector_type}"}
                
            condition_func = condition_map.get(condition.lower())
            if not condition_func:
                return {"status": "error", "message": f"Invalid condition: {condition}"}
                
            WebDriverWait(self.driver, timeout).until(
                condition_func((by_type, selector))
            )
            
            return {
                "status": "success",
                "message": f"Element with {selector_type} '{selector}' met condition '{condition}'"
            }
        except TimeoutException:
            return {"status": "timeout", "message": f"Timed out waiting for element with {selector_type}: {selector} to meet condition: {condition}"}
        except Exception as e:
            logger.error(f"Error waiting for element {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def click_element(self, selector: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Click on an element on the current page.
        
        Args:
            selector (str): The selector to find the element
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Result of the click operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        timeout = timeout or self.timeout
        selector_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }
        
        try:
            by_type = selector_map.get(selector_type.lower())
            if not by_type:
                return {"status": "error", "message": f"Invalid selector type: {selector_type}"}
                
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by_type, selector))
            )
            element.click()
            
            return {
                "status": "success",
                "message": f"Clicked element with {selector_type}: {selector}",
                "current_url": self.driver.current_url,
                "title": self.driver.title
            }
        except TimeoutException:
            return {"status": "not_found", "message": f"Element not clickable with {selector_type}: {selector}"}
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def input_text(self, selector: str, text: str, selector_type: str = "css", timeout: int = None) -> Dict[str, Any]:
        """
        Input text into an element on the current page.
        
        Args:
            selector (str): The selector to find the element
            text (str): Text to input
            selector_type (str): Type of selector ('css', 'xpath', 'id', 'class', 'name', 'tag')
            timeout (int, optional): Custom timeout for this operation
            
        Returns:
            Dict[str, Any]: Result of the text input operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        timeout = timeout or self.timeout
        selector_map = {
            "css": By.CSS_SELECTOR,
            "xpath": By.XPATH,
            "id": By.ID,
            "class": By.CLASS_NAME,
            "name": By.NAME,
            "tag": By.TAG_NAME,
        }
        
        try:
            by_type = selector_map.get(selector_type.lower())
            if not by_type:
                return {"status": "error", "message": f"Invalid selector type: {selector_type}"}
                
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by_type, selector))
            )
            element.clear()
            element.send_keys(text)
            
            return {
                "status": "success",
                "message": f"Input text into element with {selector_type}: {selector}"
            }
        except TimeoutException:
            return {"status": "not_found", "message": f"Element not found with {selector_type}: {selector}"}
        except Exception as e:
            logger.error(f"Error inputting text to element {selector}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_page_content(self) -> Dict[str, Any]:
        """
        Get the current page source, title and URL.
        
        Returns:
            Dict[str, Any]: Information about the current page
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            return {
                "status": "success",
                "title": self.driver.title,
                "url": self.driver.current_url,
                "source": self.driver.page_source
            }
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def execute_javascript(self, script: str) -> Dict[str, Any]:
        """
        Execute JavaScript on the current page.
        
        Args:
            script (str): JavaScript code to execute
            
        Returns:
            Dict[str, Any]: Result of the JavaScript execution
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            result = self.driver.execute_script(script)
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing JavaScript: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def take_screenshot(self) -> Dict[str, Any]:
        """
        Take a screenshot of the current page.
        
        Returns:
            Dict[str, Any]: Base64-encoded screenshot
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            screenshot = self.driver.get_screenshot_as_base64()
            return {
                "status": "success",
                "screenshot": screenshot
            }
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def switch_to_frame(self, frame_reference: str, reference_type: str = "index") -> Dict[str, Any]:
        """
        Switch to a frame on the page.
        
        Args:
            frame_reference (str): Reference to the frame (index, name, or ID)
            reference_type (str): Type of reference ('index', 'name', 'id', 'element')
            
        Returns:
            Dict[str, Any]: Result of the frame switch operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            if reference_type == "index":
                try:
                    index = int(frame_reference)
                    self.driver.switch_to.frame(index)
                except ValueError:
                    return {"status": "error", "message": f"Invalid frame index: {frame_reference}"}
            elif reference_type == "name" or reference_type == "id":
                self.driver.switch_to.frame(frame_reference)
            elif reference_type == "element":
                # First find the element
                selector_parts = frame_reference.split(":", 1)
                if len(selector_parts) != 2:
                    return {"status": "error", "message": "Element reference must be in format 'selector_type:selector'"}
                
                selector_type, selector = selector_parts
                element_result = self.find_element(selector, selector_type)
                
                if element_result["status"] != "success":
                    return {"status": "error", "message": f"Could not find frame element: {element_result['message']}"}
                
                # Get the actual WebElement (not just the properties)
                selector_map = {
                    "css": By.CSS_SELECTOR,
                    "xpath": By.XPATH,
                    "id": By.ID,
                    "class": By.CLASS_NAME,
                    "name": By.NAME,
                    "tag": By.TAG_NAME,
                }
                by_type = selector_map.get(selector_type.lower())
                element = self.driver.find_element(by_type, selector)
                self.driver.switch_to.frame(element)
            else:
                return {"status": "error", "message": f"Invalid reference type: {reference_type}"}
                
            return {
                "status": "success",
                "message": f"Switched to frame using {reference_type}: {frame_reference}"
            }
        except Exception as e:
            logger.error(f"Error switching to frame {frame_reference}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def switch_to_window(self, window_reference: str, reference_type: str = "index") -> Dict[str, Any]:
        """
        Switch to a window or tab.
        
        Args:
            window_reference (str): Reference to the window (index, handle, or title)
            reference_type (str): Type of reference ('index', 'handle', 'title')
            
        Returns:
            Dict[str, Any]: Result of the window switch operation
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
            
        try:
            window_handles = self.driver.window_handles
            
            if not window_handles:
                return {"status": "error", "message": "No window handles available"}
                
            if reference_type == "index":
                try:
                    index = int(window_reference)
                    if index < 0 or index >= len(window_handles):
                        return {"status": "error", "message": f"Window index out of range: {index}"}
                    
                    self.driver.switch_to.window(window_handles[index])
                except ValueError:
                    return {"status": "error", "message": f"Invalid window index: {window_reference}"}
            elif reference_type == "handle":
                if window_reference not in window_handles:
                    return {"status": "error", "message": f"Window handle not found: {window_reference}"}
                
                self.driver.switch_to.window(window_reference)
            elif reference_type == "title":
                current_handle = self.driver.current_window_handle
                window_found = False
                
                for handle in window_handles:
                    try:
                        self.driver.switch_to.window(handle)
                        if self.driver.title == window_reference:
                            window_found = True
                            break
                    except Exception:
                        pass
                
                if not window_found:
                    # Switch back to the original window
                    self.driver.switch_to.window(current_handle)
                    return {"status": "error", "message": f"No window with title '{window_reference}' found"}
            else:
                return {"status": "error", "message": f"Invalid reference type: {reference_type}"}
                
            return {
                "status": "success",
                "message": f"Switched to window using {reference_type}: {window_reference}",
                "title": self.driver.title,
                "url": self.driver.current_url
            }
        except Exception as e:
            logger.error(f"Error switching to window {window_reference}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_tool_descriptions(self) -> List[str]:
        """
        Returns a brief description of the browser tool.
        
        Returns:
            List[str]: Tool descriptions
        """
        return [
            "Interact with web browsers to navigate, click, input text, and extract information from web pages."
        ]

    def get_cookies(self) -> Dict[str, Any]:
        """Get all cookies from the browser"""
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        try:
            cookies = self.driver.get_cookies()
            return {"status": "success", "cookies": cookies}
        except Exception as e:
            logger.error(f"Error getting cookies: {str(e)}")
            return {"status": "error", "message": str(e)}

    def add_cookie(self, cookie_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add a cookie to the browser"""
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        try:
            self.driver.add_cookie(cookie_dict)
            return {"status": "success", "message": "Cookie added successfully"}
        except Exception as e:
            logger.error(f"Error adding cookie: {str(e)}")
            return {"status": "error", "message": str(e)}

    def select_dropdown_option(self, select_selector: str, 
                              option_value: str,
                              select_by: str = "value",
                              selector_type: str = "css") -> Dict[str, Any]:
        """
        Select an option from a dropdown
        select_by can be 'value', 'text', or 'index'
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        
        try:
            from selenium.webdriver.support.ui import Select
            # Find the dropdown element
            element_result = self.find_element(select_selector, selector_type)
            if element_result["status"] != "success":
                return element_result
            
            # Get the actual element
            selector_map = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "class": By.CLASS_NAME,
                "name": By.NAME,
                "tag": By.TAG_NAME,
            }
            by_type = selector_map.get(selector_type.lower())
            element = self.driver.find_element(by_type, select_selector)
            
            # Create select object
            select = Select(element)
            
            # Select based on method
            if select_by.lower() == "value":
                select.select_by_value(option_value)
            elif select_by.lower() == "text":
                select.select_by_visible_text(option_value)
            elif select_by.lower() == "index":
                select.select_by_index(int(option_value))
            else:
                return {"status": "error", "message": f"Invalid select_by option: {select_by}"}
            
            return {"status": "success", "message": f"Selected option with {select_by}: {option_value}"}
        except Exception as e:
            logger.error(f"Error selecting dropdown option: {str(e)}")
            return {"status": "error", "message": str(e)}

    def close_browser(self) -> Dict[str, Any]:
        """
        Close the browser and end the session.
        
        Returns:
            Dict[str, Any]: Status of the browser closure
        """
        if not self.driver:
            return {"status": "success", "message": "Browser already closed"}
            
        try:
            self.driver.quit()
            self.driver = None
            return {"status": "success", "message": "Browser closed successfully"}
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_tools(self) -> List[Callable]:
        """
        Returns a list of callable functions for all tools
        
        Returns:
            List[Callable]: A list of callable functions
        """
        # IMPORTANT: This order MUST match the exact order of schemas in get_tool_schemas()
        return [
            self.initialize_browser,
            self.navigate_to_url,
            self.find_element,
            self.find_multiple_elements,
            self.wait_for_element,
            self.click_element,
            self.input_text,
            self.select_dropdown_option, 
            self.get_page_content,
            self.execute_javascript,
            self.take_screenshot,
            self.switch_to_frame,
            self.switch_to_window,
            self.get_cookies,
            self.add_cookie,
            self.close_browser
        ]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for browser tools.
        
        Returns:
            List[Dict[str, Any]]: Function schemas in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "initialize_browser",
                    "description": "Initialize and start a new browser session.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "navigate_to_url",
                    "description": "Navigate the browser to a specific URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to navigate to"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout for this operation in seconds"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_element",
                    "description": "Find an element on the current page and return information about it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The selector to find the element"
                            },
                            "selector_type": {
                                "type": "string",
                                "description": "Type of selector (css, xpath, id, class, name, tag)",
                                "enum": ["css", "xpath", "id", "class", "name", "tag"]
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout for this operation in seconds"
                            }
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_multiple_elements",
                    "description": "Find multiple elements on the current page and return information about them.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The selector to find the elements"
                            },
                            "selector_type": {
                                "type": "string",
                                "description": "Type of selector (css, xpath, id, class, name, tag)",
                                "enum": ["css", "xpath", "id", "class", "name", "tag"]
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout for this operation in seconds"
                            }
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "wait_for_element",
                    "description": "Wait for an element to meet a specific condition.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The selector to find the element"
                            },
                            "condition": {
                                "type": "string",
                                "description": "The condition to wait for",
                                "enum": ["presence", "visibility", "clickable", "invisibility"]
                            },
                            "selector_type": {
                                "type": "string",
                                "description": "Type of selector (css, xpath, id, class, name, tag)",
                                "enum": ["css", "xpath", "id", "class", "name", "tag"]
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout for this operation in seconds"
                            }
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "click_element",
                    "description": "Click on an element on the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The selector to find the element"
                            },
                            "selector_type": {
                                "type": "string",
                                "description": "Type of selector (css, xpath, id, class, name, tag)",
                                "enum": ["css", "xpath", "id", "class", "name", "tag"]
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout for this operation in seconds"
                            }
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "input_text",
                    "description": "Input text into an element on the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "The selector to find the element"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to input"
                            },
                            "selector_type": {
                                "type": "string",
                                "description": "Type of selector (css, xpath, id, class, name, tag)",
                                "enum": ["css", "xpath", "id", "class", "name", "tag"]
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Custom timeout for this operation in seconds"
                            }
                        },
                        "required": ["selector", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_dropdown_option",
                    "description": "Select an option from a dropdown menu.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "select_selector": {
                                "type": "string",
                                "description": "The selector to find the dropdown element"
                            },
                            "option_value": {
                                "type": "string",
                                "description": "The value or text of the option to select"
                            },
                            "select_by": {
                                "type": "string",
                                "description": "How to select the option (by value, visible text, or index)",
                                "enum": ["value", "text", "index"]
                            },
                            "selector_type": {
                                "type": "string",
                                "description": "Type of selector for the dropdown element",
                                "enum": ["css", "xpath", "id", "class", "name", "tag"]
                            }
                        },
                        "required": ["select_selector", "option_value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_content",
                    "description": "Get the current page source, title and URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_javascript",
                    "description": "Execute JavaScript on the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "JavaScript code to execute"
                            }
                        },
                        "required": ["script"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": "Take a screenshot of the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "switch_to_frame",
                    "description": "Switch to a frame on the page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "frame_reference": {
                                "type": "string",
                                "description": "Reference to the frame (index, name, or ID)"
                            },
                            "reference_type": {
                                "type": "string",
                                "description": "Type of reference",
                                "enum": ["index", "name", "id", "element"]
                            }
                        },
                        "required": ["frame_reference"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "switch_to_window",
                    "description": "Switch to a window or tab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "window_reference": {
                                "type": "string",
                                "description": "Reference to the window (index, handle, or title)"
                            },
                            "reference_type": {
                                "type": "string",
                                "description": "Type of reference",
                                "enum": ["index", "handle", "title"]
                            }
                        },
                        "required": ["window_reference"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cookies",
                    "description": "Get all cookies from the browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_cookie",
                    "description": "Add a cookie to the browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cookie_dict": {
                                "type": "object",
                                "description": "Cookie dictionary with name, value, etc.",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Cookie name"
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "Cookie value"
                                    },
                                    "domain": {
                                        "type": "string",
                                        "description": "Cookie domain (optional)"
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "Cookie path (optional)"
                                    },
                                    "secure": {
                                        "type": "boolean",
                                        "description": "Whether cookie is secure (optional)"
                                    },
                                    "expiry": {
                                        "type": "integer",
                                        "description": "Cookie expiry timestamp (optional)"
                                    }
                                },
                                "required": ["name", "value"]
                            }
                        },
                        "required": ["cookie_dict"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "close_browser",
                    "description": "Close the browser and end the session.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
    
    def get_tool_prompt(self) -> str:
        """
        Returns a tool using instruction prompt for agent to use the browser tool.
        
        Returns:
            str: Tool description
        """
        return """
You are a web browser agent with the ability to interact with web pages. Follow these guidelines to effectively use browser tools:

OBSERVATION:
- After navigating to a page, first use get_page_content() to understand the overall structure
- Use find_element() and find_multiple_elements() to locate specific components
- When elements aren't found, try different selector types (css, xpath, id) or refine your selector
- Wait for elements with wait_for_element() before interacting with dynamic content

EXTRACTION:
- Extract text from elements by finding them first, then accessing their text properties
- For structured data, identify patterns in element containers and iterate through collections
- Use execute_javascript() for complex data extraction when simple selectors aren't sufficient
- Take screenshots when visual context would be helpful for analysis

NAVIGATION:
- Start with navigate_to_url() to access a website
- Use click_element() for links, buttons, and interactive elements
- Handle forms with input_text() for text fields and select_dropdown_option() for dropdowns
- Navigate between frames with switch_to_frame() when content is embedded
- Use switch_to_window() when actions open new tabs or windows

GENERAL GUIDELINES:
- Only make one tool call at a time
- After each action, verify the page state with get_page_content() or finding relevant elements
- Handle timeouts and errors by adjusting selectors or waiting for elements
- Close the browser with close_browser() when finished

Start by initializing the browser with initialize_browser() before performing any other operations.
"""
        
