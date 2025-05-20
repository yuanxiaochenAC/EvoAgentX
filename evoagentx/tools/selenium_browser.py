#!/usr/bin/env python3

import time
import base64
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    ElementNotInteractableException
)
from selenium.webdriver.remote.webelement import WebElement
from .tool import Tool


class SeleniumBrowserTool(Tool):
    """
    A selenium-based browser automation tool that provides functionality similar to Playwright.
    This tool is designed to integrate with the evoagentx tools framework.
    """
    
    def __init__(self, 
                headless: bool = False, 
                browser_name: str = "chrome",
                timeout: int = 30,
                window_size: Tuple[int, int] = (1920, 1080),
                name: str = "Selenium Browser Tool",
                schemas: Optional[List[dict]] = None,
                descriptions: Optional[List[str]] = None,
                tools: Optional[List[Callable]] = None,
                **kwargs):
        """
        Initialize the browser tool with the specified options.
        
        Args:
            headless: Whether to run the browser in headless mode
            browser_name: The browser to use (currently only chrome is supported)
            timeout: Default timeout for browser operations in seconds
            window_size: Browser window dimensions as (width, height)
            name: Name of the tool
            schemas: Tool schemas
            descriptions: Tool descriptions
            tools: Tool functions
            **kwargs: Additional keyword arguments for parent class initialization
        """
        # Set default values if not provided
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
        
        self.browser_name = browser_name
        self.timeout = timeout
        self.window_width, self.window_height = window_size
        
        # Configure browser options
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument(f"--window-size={self.window_width},{self.window_height}")
        
        # Enable console logs collection
        self.options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        
        # Initialize the driver
        self.driver = webdriver.Chrome(options=self.options)
        
        # Store console messages
        self.console_messages = []
        
        # Tab information
        self.tabs = []
        self.current_tab_index = 0
        
        # Initialize first tab
        self.tabs.append(self.driver.current_window_handle)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for this tool.
        
        Returns:
            List[Dict[str, Any]]: The function schema in OpenAI format
        """
        return [{
            "name": "browser_navigate",
            "description": "Navigate to a specified URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        }, {
            "name": "browser_type",
            "description": "Type text into a specified element",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Human-readable element description"
                    },
                    "ref": {
                        "type": "string",
                        "description": "CSS selector or XPath to locate the element"
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type"
                    },
                    "submit": {
                        "type": "boolean",
                        "description": "Whether to press Enter after typing",
                        "default": False
                    },
                    "slowly": {
                        "type": "boolean",
                        "description": "Whether to type one character at a time",
                        "default": False
                    }
                },
                "required": ["element", "ref", "text"]
            }
        }, {
            "name": "browser_click",
            "description": "Click on an element",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Human-readable element description"
                    },
                    "ref": {
                        "type": "string",
                        "description": "CSS selector or XPath to locate the element"
                    }
                },
                "required": ["element", "ref"]
            }
        }, {
            "name": "browser_snapshot",
            "description": "Take a snapshot of the current page, including its structure",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }, {
            "name": "browser_console_messages",
            "description": "Get console messages from the browser",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }]
    
    def get_tools(self) -> List[Callable]:
        """
        Returns a list of callable functions for all tools
        
        Returns:
            List[Callable]: A list of callable functions
        """
        return [
            self.browser_navigate,
            self.browser_type,
            self.browser_click,
            self.browser_snapshot,
            self.browser_console_messages
        ]
    
    def get_tool_descriptions(self) -> List[str]:
        """
        Returns a list of descriptions for all tools
        
        Returns:
            List[str]: A list of descriptions
        """
        return [
            "A selenium-based browser automation tool that provides functionality for web browsing, page interaction, and content extraction."
        ]
    
    def _wait_for_element(self, locator: Tuple[By, str], timeout: Optional[int] = None) -> WebElement:
        """
        Wait for an element to be present and return it.
        
        Args:
            locator: A tuple of (By, selector)
            timeout: Maximum time to wait for the element (uses default if None)
            
        Returns:
            The WebElement found
            
        Raises:
            TimeoutException if the element is not found within the timeout
        """
        timeout = timeout or self.timeout
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located(locator)
        )
    
    def _wait_for_clickable(self, locator: Tuple[By, str], timeout: Optional[int] = None) -> WebElement:
        """
        Wait for an element to be clickable and return it.
        
        Args:
            locator: A tuple of (By, selector)
            timeout: Maximum time to wait for the element (uses default if None)
            
        Returns:
            The WebElement found
            
        Raises:
            TimeoutException if the element is not clickable within the timeout
        """
        timeout = timeout or self.timeout
        return WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )
    
    def browser_navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate to the specified URL.
        
        Args:
            url: The URL to navigate to
            
        Returns:
            A dict with information about the navigation
        """
        try:
            self.driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, self.timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Collect page information
            title = self.driver.title
            current_url = self.driver.current_url
            
            return {
                "url": current_url,
                "title": title,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to navigate to {url}",
                "error": str(e)
            }
    
    def browser_type(self, 
                    element: str, 
                    ref: str, 
                    text: str, 
                    submit: bool = False, 
                    slowly: bool = False) -> Dict[str, Any]:
        """
        Type text into an element.
        
        Args:
            element: Human-readable element description
            ref: CSS selector or XPath to locate the element
            text: The text to type
            submit: Whether to press Enter after typing
            slowly: Whether to type one character at a time
            
        Returns:
            A dict with information about the typing action
        """
        try:
            # Determine if ref is XPath or CSS selector
            if ref.startswith("//") or ref.startswith("(//"):
                elem = self._wait_for_element((By.XPATH, ref))
            else:
                elem = self._wait_for_element((By.CSS_SELECTOR, ref))
            
            # Focus on the element
            elem.click()
            
            # Clear existing content
            elem.clear()
            
            # Type text based on slowly parameter
            if slowly:
                for char in text:
                    elem.send_keys(char)
                    time.sleep(0.1)
            else:
                elem.send_keys(text)
            
            # Submit if requested
            if submit:
                elem.send_keys(Keys.ENTER)
            
            return {
                "status": "success",
                "element": element,
                "text": text
            }
        
        except (TimeoutException, NoSuchElementException) as e:
            return {
                "status": "error",
                "message": f"Element not found: {ref}",
                "error": str(e)
            }
        except ElementNotInteractableException as e:
            return {
                "status": "error",
                "message": f"Element not interactable: {ref}",
                "error": str(e)
            }
    
    def browser_click(self, element: str, ref: str) -> Dict[str, Any]:
        """
        Click on an element.
        
        Args:
            element: Human-readable element description
            ref: CSS selector or XPath to locate the element
            
        Returns:
            A dict with information about the click action
        """
        try:
            # Determine if ref is XPath or CSS selector
            if ref.startswith("//") or ref.startswith("(//"):
                elem = self._wait_for_clickable((By.XPATH, ref))
            else:
                elem = self._wait_for_clickable((By.CSS_SELECTOR, ref))
            
            # Scroll the element into view
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
            
            # Click the element
            elem.click()
            
            # Wait for any potential page load
            WebDriverWait(self.driver, 5).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            return {
                "status": "success",
                "element": element,
                "url": self.driver.current_url,
                "title": self.driver.title
            }
        
        except (TimeoutException, NoSuchElementException) as e:
            return {
                "status": "error",
                "message": f"Element not found: {ref}",
                "error": str(e)
            }
        except ElementNotInteractableException as e:
            return {
                "status": "error",
                "message": f"Element not interactable: {ref}",
                "error": str(e)
            }
    
    def browser_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of the current page, including its structure.
        
        Returns:
            A dict with information about the page snapshot
        """
        try:
            # Get page information
            url = self.driver.current_url
            title = self.driver.title
            
            # Create a simple representation of page elements (form fields, buttons, links)
            elements = []
            
            # Find all interactive elements
            for element_type, selector in [
                ("button", "button, input[type='button'], input[type='submit']"),
                ("link", "a"),
                ("textbox", "input[type='text'], input[type='password'], input[type='email'], textarea"),
                ("checkbox", "input[type='checkbox']"),
                ("select", "select"),
            ]:
                try:
                    elems = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for idx, elem in enumerate(elems):
                        try:
                            text = elem.text.strip() or elem.get_attribute("value") or elem.get_attribute("placeholder")
                            if not text and element_type == "link":
                                text = elem.get_attribute("href")
                            elif not text:
                                text = f"{element_type} {idx+1}"
                            
                            elements.append({
                                "type": element_type,
                                "text": text,
                                "ref": self._generate_selector(elem),
                            })
                        except:
                            # Skip elements that cause errors
                            pass
                except:
                    # Skip element types that cause errors
                    pass
            
            # Take a screenshot
            screenshot = self.driver.get_screenshot_as_base64()
            
            return {
                "url": url,
                "title": title,
                "elements": elements,
                "screenshot": screenshot,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": "Failed to take snapshot",
                "error": str(e)
            }
    
    def browser_console_messages(self) -> Dict[str, Any]:
        """
        Get console messages from the browser.
        
        Returns:
            A dict with console messages
        """
        try:
            # Collect console logs
            logs = self.driver.get_log('browser')
            
            # Format the logs
            formatted_logs = []
            for log in logs:
                level = log.get('level', '').upper()
                message = log.get('message', '')
                
                # Clean up message format, often has "console-api" and other noise
                if 'console-api' in message and ':' in message:
                    message = message.split(':', 1)[1].strip()
                
                # Skip non-console messages
                if level not in ['INFO', 'WARNING', 'ERROR']:
                    continue
                    
                # Map level to LOG, WARN, ERROR
                if level == 'INFO':
                    log_type = 'LOG'
                elif level == 'WARNING':
                    log_type = 'WARN'
                else:
                    log_type = level
                    
                formatted_logs.append(f"[{log_type}] {message}")
            
            return {
                "messages": formatted_logs,
                "text": '\n'.join(formatted_logs),
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": "Failed to retrieve console messages",
                "error": str(e)
            }
    
    def execute(self, action: str, **params) -> Dict[str, Any]:
        """
        Generic execution method that dispatches to the appropriate browser tool method.
        This method conforms to the standard tool interface pattern.
        
        Args:
            action: The browser action to perform (navigate, click, type, snapshot, console_messages)
            **params: Parameters specific to the action
            
        Returns:
            A dict with the result of the operation
        """
        try:
            if action == "navigate":
                return self.browser_navigate(params["url"])
            elif action == "click":
                return self.browser_click(params["element"], params["ref"])
            elif action == "type":
                return self.browser_type(
                    params["element"], 
                    params["ref"], 
                    params["text"], 
                    params.get("submit", False), 
                    params.get("slowly", False)
                )
            elif action == "snapshot":
                return self.browser_snapshot()
            elif action == "console_messages":
                return self.browser_console_messages()
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
        except KeyError as e:
            return {
                "status": "error",
                "message": f"Missing required parameter: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to execute {action}",
                "error": str(e)
            }
    
    def _generate_selector(self, element: WebElement) -> str:
        """
        Generate a unique CSS selector or XPath for an element.
        
        Args:
            element: The WebElement to generate a selector for
            
        Returns:
            A CSS selector or XPath that uniquely identifies the element
        """
        # Try to use ID if available
        element_id = element.get_attribute("id")
        if element_id:
            return f"#{element_id}"
        
        # Try to use a class if available
        element_class = element.get_attribute("class")
        if element_class:
            classes = element_class.split()
            if classes:
                # Use the first class
                return f".{classes[0]}"
        
        # Generate an XPath as a fallback
        try:
            # This is a simplified approach, a more robust implementation would use 
            # a library or more complex logic to generate a reliable XPath
            tag_name = element.tag_name
            element_text = element.text.strip() if element.text else None
            
            if element_text:
                return f"//{tag_name}[contains(text(), '{element_text}')]"
            else:
                # Generate an XPath based on its position in the DOM
                script = """
                function getPathTo(element) {
                    if (element.id && document.getElementById(element.id) === element)
                        return '//*[@id="' + element.id + '"]';
                    
                    if (element === document.body)
                        return '/html/body';
                        
                    var index = 1;
                    var siblings = element.parentNode.childNodes;
                    
                    for (var i = 0; i < siblings.length; i++) {
                        var sibling = siblings[i];
                        if (sibling === element)
                            return getPathTo(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + index + ']';
                        if (sibling.nodeType === 1 && sibling.tagName === element.tagName)
                            index++;
                    }
                }
                return getPathTo(arguments[0]);
                """
                return self.driver.execute_script(script, element)
        except:
            # Return a basic XPath if all else fails
            return f"//{element.tag_name}"
    
    def close(self):
        """Close the browser and clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None

