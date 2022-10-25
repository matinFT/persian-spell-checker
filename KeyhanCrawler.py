from selenium import webdriver
import json
from os.path import exists
import time


def get_page_posts(driver: webdriver, url: str):
    driver.get(url)
    time.sleep(4)
    posts = {}
    post_elements = driver.find_elements_by_css_selector("div.archive_content > div.linear_news > a")
    for post in post_elements:
        link, title = None, None
        try:
            title = post.text
            link = post.get_attribute("href")
            id = link.split("/")[5]
            posts[id] = {
                "link": link,
                "title": title
            }
        except Exception as e:
            print("post elements", link)
            print(e)

    for key in posts:
        i = 1
        while i <= 3:
            try:
                driver.get(posts[key]["link"])
                time.sleep(i*2 - 1)
                content = driver.find_element_by_css_selector("div.body_news").text
                posts[key]["content"] = content
                break
            except Exception as e:
                print(f'contents {i}', posts[key]["link"])
                print(e)
                posts[key]["content"] = "Error"
                i+=1
        if i == 4:
            time.sleep(60)

    return posts


options = webdriver.FirefoxOptions()
driver = webdriver.Firefox(options=options)
posts_url_page = "https://kayhan.ir/fa/archive?service_id=1&sec_id=&cat_id=&rpp=100&from_date=1392/07/06&to_date=1401/07/21&p="
if exists("Keyhan.json"):
    posts = json.load(open("Keyhan.json", "r"))
else:
    posts = {}
temp_posts = {}
for i in range(581, 651):
    try:
        temp_posts = get_page_posts(driver, posts_url_page + str(i))
    except Exception as e:
        print(f'error happened in fetching page {i}/2505')
        print(e)
    posts.update(temp_posts)
    print(f'page {i}/2505 finished')


with open('Keyhan.json', 'w', encoding='utf-8') as outfile:
    json.dump(posts, outfile)

# 1 to 650 except 564, 565
