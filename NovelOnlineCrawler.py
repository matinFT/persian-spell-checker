from selenium import webdriver
import json
from os.path import exists
import time


driver = webdriver.Chrome(executable_path='E:\Downloads\Compressed\chromedriver.exe')
posts_url_page = "https://novelonline.ir/online"
driver.get(posts_url_page)
novels = dict.fromkeys([x.get_attribute("href") for x in driver.find_elements_by_css_selector("div.masonry > div.free > div.panel > div.panel-body > a")], {})

for novel in novels:
    driver.get(novel)
    parts_row = driver.find_elements_by_css_selector("div.content > div.row > div.col-md-8 > div.row")[3]
    try:
        more_part_button = parts_row.find_element_by_css_selector("div.col-md-12 > button")
        more_part_button.click()
    except Exception as e:
        print(e)
    novels[novel] = dict.fromkeys([x.get_attribute("href") for x in parts_row.find_elements_by_css_selector("div.col-md-6 > div.card > div.card-body > p > a")], "")
    for part in novels[novel]:
        driver.get(part)
        content = driver.find_elements_by_css_selector("div.content > div.row > div.col-md-8 > div.row")[0].find_element_by_css_selector("div.panel-body > blockquote").text
        novels[novel][part] = content

with open('NovelOnline.json', 'w', encoding='utf-8') as outfile:
    json.dump(novels, outfile)
