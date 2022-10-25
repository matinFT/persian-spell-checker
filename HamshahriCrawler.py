from selenium import webdriver
import json
from os.path import exists
import time


def get_page_newspapers(driver, page_url):
    driver.get(page_url)
    panel_newspapers = driver.find_elements_by_css_selector("div.panel-body > div.col-md-4 > div.newspaper > a")
    newspapers = {}
    for newspaper in panel_newspapers:
        newspaper_link = None
        try:
            newspaper_number = newspaper.find_element_by_css_selector("span").text
            newspaper_date = newspaper.get_attribute('title')
            newspaper_link = newspaper.get_attribute('href')
            newspapers[newspaper_number] = {
                "date": newspaper_date,
                "link": newspaper_link
            }
        except Exception as e:
            print(1, e, newspaper_link)
        # break

    for key in newspapers:
        newspapers[key]["posts"] = get_newspaper_posts(driver, newspapers[key]["link"])
    return newspapers


def get_newspaper_posts(driver, newspaper_url):
    driver.get(newspaper_url)
    newspaper = {}
    newspaper_posts = driver.find_elements_by_css_selector("div.post")
    for post in newspaper_posts:
        try:
            post_link = post.find_element_by_css_selector("a").get_attribute('href')
            for i in range(1, 6):
                try:
                    post_header = post.find_element_by_css_selector(f'h{i}').text
                    newspaper[post_header] = post_link
                    break
                except:
                    pass
        except:
            pass
    for key in newspaper:
        i = 1
        while i <= 3:
            try:
                driver.get(newspaper[key])
                time.sleep(i*2 - 1)
                newspaper[key] = driver.find_element_by_css_selector("div.news-main-content > div.text").text
                break
            except Exception as e:
                print(f'posts {i}', e)
                i += 1
                pass
        if i == 4:
            print("unable to get post with following link ", newspaper[key])
            newspaper[key] = "error"
            time.sleep(60)
    return newspaper


options = webdriver.FirefoxOptions()
driver = webdriver.Firefox(options=options)
yearly_newspapers_page = [
    # "https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1401&newspapermonth=0&newspaperday=0",
    # "https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1400&newspapermonth=0&newspaperday=0",
    # "https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1399&newspapermonth=0&newspaperday=0",
    # "https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1398&newspapermonth=0&newspaperday=0",
    # "https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1397&newspapermonth=0&newspaperday=0",
    # "https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1396&newspapermonth=0&newspaperday=0"
    ]

years_to_crawl = [1397, 1396]

if exists("Hamshahri.json"):
    all_newspapers = json.load(open("Hamshahri.json", "r"))
else:
    all_newspapers = {}
page_newspapers = {}

for year in years_to_crawl:
    for i in range(1, 13):
        try:
            url = f'https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear={year}&newspapermonth={i}&newspaperday=0'
            page_newspapers = get_page_newspapers(driver, url)
            # url="https://newspaper.hamshahrionline.ir/archive/page/778-02?newspaperyear=1401&newspapermonth=01&newspaperday=0"
            # this_year_newspapers = get_this_year_newspapers(driver, url)
            all_newspapers.update(page_newspapers)
        except Exception as e:
            print(3, e)

with open('Hamshahri.json', 'w') as outfile:
    json.dump(all_newspapers, outfile)

# 1401, 1400, 1399, 1398
