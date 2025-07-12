"""Browser automation functionality."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from ..utils.config import Config
from ..utils.exceptions import AutomationError
from ..utils.logger import get_automation_logger

logger = get_automation_logger()


class BrowserAutomation:
    """Browser automation using Selenium."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize browser automation.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config.load_default()
        self.browser_config = self.config.automation.browser
        
        self.driver: Optional[webdriver.Remote] = None
        self.wait: Optional[WebDriverWait] = None
        
        # Browser settings
        self.default_browser = self.browser_config.get("default", "chrome")
        self.headless = self.browser_config.get("headless", False)
        self.timeout = self.browser_config.get("timeout", 30)
        self.window_size = self.browser_config.get("window_size", "1920x1080")
    
    async def start_browser(
        self,
        browser: Optional[str] = None,
        headless: Optional[bool] = None,
        user_data_dir: Optional[str] = None
    ) -> bool:
        """
        Start browser instance.
        
        Args:
            browser: Browser type (chrome, firefox, safari, edge)
            headless: Run in headless mode
            user_data_dir: User data directory for persistent sessions
            
        Returns:
            True if successful
        """
        browser = browser or self.default_browser
        headless = headless if headless is not None else self.headless
        
        try:
            if browser.lower() == "chrome":
                self.driver = self._create_chrome_driver(headless, user_data_dir)
            elif browser.lower() == "firefox":
                self.driver = self._create_firefox_driver(headless, user_data_dir)
            else:
                raise AutomationError(f"Unsupported browser: {browser}")
            
            # Set window size
            if not headless:
                width, height = map(int, self.window_size.split("x"))
                self.driver.set_window_size(width, height)
            
            # Initialize WebDriverWait
            self.wait = WebDriverWait(self.driver, self.timeout)
            
            logger.info(f"Started {browser} browser (headless: {headless})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False
    
    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if successful
        """
        if not self.driver:
            raise AutomationError("Browser not started")
        
        try:
            self.driver.get(url)
            logger.info(f"Navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return False
    
    async def find_element(
        self,
        selector: str,
        by: str = "css",
        timeout: Optional[int] = None
    ) -> Optional[Any]:
        """
        Find element on page.
        
        Args:
            selector: Element selector
            by: Selection method (css, xpath, id, name, class, tag)
            timeout: Wait timeout
            
        Returns:
            WebElement or None if not found
        """
        if not self.driver:
            raise AutomationError("Browser not started")
        
        timeout = timeout or self.timeout
        
        try:
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME,
            }
            
            by_method = by_mapping.get(by.lower(), By.CSS_SELECTOR)
            
            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(EC.presence_of_element_located((by_method, selector)))
            
            logger.debug(f"Found element: {selector}")
            return element
            
        except Exception as e:
            logger.warning(f"Element not found: {selector} ({e})")
            return None
    
    async def click_element(
        self,
        selector: str,
        by: str = "css",
        timeout: Optional[int] = None
    ) -> bool:
        """
        Click an element.
        
        Args:
            selector: Element selector
            by: Selection method
            timeout: Wait timeout
            
        Returns:
            True if successful
        """
        element = await self.find_element(selector, by, timeout)
        if not element:
            return False
        
        try:
            # Wait for element to be clickable
            wait = WebDriverWait(self.driver, timeout or self.timeout)
            clickable_element = wait.until(EC.element_to_be_clickable(element))
            clickable_element.click()
            
            logger.info(f"Clicked element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Failed to click element {selector}: {e}")
            return False
    
    async def type_in_element(
        self,
        selector: str,
        text: str,
        by: str = "css",
        clear_first: bool = True,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Type text in an element.
        
        Args:
            selector: Element selector
            text: Text to type
            by: Selection method
            clear_first: Clear element before typing
            timeout: Wait timeout
            
        Returns:
            True if successful
        """
        element = await self.find_element(selector, by, timeout)
        if not element:
            return False
        
        try:
            if clear_first:
                element.clear()
            
            element.send_keys(text)
            logger.info(f"Typed text in element {selector}: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to type in element {selector}: {e}")
            return False
    
    async def get_element_text(
        self,
        selector: str,
        by: str = "css",
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Get text from an element.
        
        Args:
            selector: Element selector
            by: Selection method
            timeout: Wait timeout
            
        Returns:
            Element text or None
        """
        element = await self.find_element(selector, by, timeout)
        if not element:
            return None
        
        try:
            text = element.text
            logger.debug(f"Got text from element {selector}: {text[:50]}...")
            return text
        except Exception as e:
            logger.error(f"Failed to get text from element {selector}: {e}")
            return None
    
    async def scroll_to_element(
        self,
        selector: str,
        by: str = "css",
        timeout: Optional[int] = None
    ) -> bool:
        """
        Scroll to an element.
        
        Args:
            selector: Element selector
            by: Selection method
            timeout: Wait timeout
            
        Returns:
            True if successful
        """
        element = await self.find_element(selector, by, timeout)
        if not element:
            return False
        
        try:
            self.driver.execute_script("arguments[0].scrollIntoView();", element)
            logger.info(f"Scrolled to element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Failed to scroll to element {selector}: {e}")
            return False
    
    async def wait_for_page_load(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for page to load completely.
        
        Args:
            timeout: Wait timeout
            
        Returns:
            True if page loaded
        """
        if not self.driver:
            return False
        
        timeout = timeout or self.timeout
        
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
            logger.debug("Page loaded completely")
            return True
        except Exception as e:
            logger.warning(f"Page load timeout: {e}")
            return False
    
    async def take_screenshot(self, filepath: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Take a screenshot of the current page.
        
        Args:
            filepath: Path to save screenshot
            
        Returns:
            Path to screenshot file or None if failed
        """
        if not self.driver:
            return None
        
        try:
            if filepath is None:
                filepath = Path(f"screenshot_{int(time.time())}.png")
            else:
                filepath = Path(filepath)
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            self.driver.save_screenshot(str(filepath))
            logger.info(f"Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    async def execute_javascript(self, script: str) -> Any:
        """
        Execute JavaScript in the browser.
        
        Args:
            script: JavaScript code to execute
            
        Returns:
            Script result
        """
        if not self.driver:
            raise AutomationError("Browser not started")
        
        try:
            result = self.driver.execute_script(script)
            logger.debug(f"Executed JavaScript: {script[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Failed to execute JavaScript: {e}")
            return None
    
    async def get_current_url(self) -> Optional[str]:
        """
        Get current page URL.
        
        Returns:
            Current URL or None
        """
        if not self.driver:
            return None
        
        try:
            return self.driver.current_url
        except Exception as e:
            logger.error(f"Failed to get current URL: {e}")
            return None
    
    async def get_page_title(self) -> Optional[str]:
        """
        Get current page title.
        
        Returns:
            Page title or None
        """
        if not self.driver:
            return None
        
        try:
            return self.driver.title
        except Exception as e:
            logger.error(f"Failed to get page title: {e}")
            return None
    
    async def close_browser(self) -> None:
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
            finally:
                self.driver = None
                self.wait = None
    
    def _create_chrome_driver(
        self,
        headless: bool,
        user_data_dir: Optional[str]
    ) -> webdriver.Chrome:
        """Create Chrome WebDriver instance."""
        options = ChromeOptions()
        
        if headless:
            options.add_argument("--headless")
        
        # Standard Chrome options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        
        # User data directory for persistent sessions
        if user_data_dir:
            options.add_argument(f"--user-data-dir={user_data_dir}")
        
        # Add configured options
        chrome_options = self.browser_config.get("chrome_options", [])
        for option in chrome_options:
            options.add_argument(option)
        
        return webdriver.Chrome(options=options)
    
    def _create_firefox_driver(
        self,
        headless: bool,
        user_data_dir: Optional[str]
    ) -> webdriver.Firefox:
        """Create Firefox WebDriver instance."""
        options = FirefoxOptions()
        
        if headless:
            options.add_argument("--headless")
        
        # Add configured options
        firefox_options = self.browser_config.get("firefox_options", [])
        for option in firefox_options:
            options.add_argument(option)
        
        # Profile directory
        if user_data_dir:
            profile = webdriver.FirefoxProfile(user_data_dir)
            return webdriver.Firefox(firefox_profile=profile, options=options)
        
        return webdriver.Firefox(options=options)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()
