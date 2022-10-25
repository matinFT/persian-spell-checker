from selenium import webdriver
import json
from os.path import exists
import time

options = webdriver.FirefoxOptions()
driver = webdriver.Firefox(options=options)
driver.get("https://fa.wikipedia.org/wiki/%D8%AC%D8%A7%D9%85_%D8%AC%D9%85_(%D8%B1%D9%88%D8%B2%D9%86%D8%A7%D9%85%D9%87)")
b = driver.find_element_by_css_selector("body")
print(b.text)
