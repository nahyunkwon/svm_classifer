from selenium import webdriver
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import StaleElementReferenceException
import time
import pandas as pd
import requests
from requests.exceptions import SSLError
from urllib3.exceptions import MaxRetryError
import re
import os
import pathlib


def get_image_urls():
    driver = webdriver.Chrome('./chromedriver')
    driver2 = webdriver.Chrome('./chromedriver')
    driver.implicitly_wait(5)

    driver.get('https://www.google.com/search?q=3d+printing+supports&tbm=isch&ved=2ahUKEwjX3KiFkPHsAhVDbK0KHeYUDBMQ2-cCegQIABAA&oq=3d+printing+supports&gs_lcp=CgNpbWcQAzIECAAQQzICCAAyAggAMgYIABAFEB4yBggAEAUQHjIGCAAQBRAeMgYIABAIEB4yBAgAEBgyBAgAEBgyBAgAEBhQqOkYWKjpGGD26hhoAHAAeACAATWIATWSAQExmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=zu-mX9e9JMPYtQXmqbCYAQ&bih=1079&biw=1904&rlz=1C1CHBF_enUS922US922')
    result = []
    related_images_url = []

    file_name = "support"

    time.sleep(3)

    for i in range(0, 200):
        try:
            thumbnail_image = driver.find_elements_by_class_name("rg_i")[i]
            thumbnail_image.click()
            image = driver.find_elements_by_css_selector(".n3VNCb")

            time.sleep(10)
            result.clear()

            for k in image:
                src = k.get_attribute("src")
                # avoid scraping thumbnail images-stringing-support below the target image
                if "http" in src:
                    if "jpg" in src or "jpeg" in src or "png" in src:

                        show_more = driver.find_elements_by_css_selector(".So4Urb")[0]
                        related_url = show_more.get_attribute('href')

                        result.append([str(i), src, related_url])
                        print(result)

                        driver2.get(related_url)

                        for j in range(0, 20):

                            thumbnail_image_related = driver2.find_elements_by_class_name("rg_i")[j]
                            thumbnail_image_related.click()
                            related_images = driver2.find_elements_by_css_selector(".n3VNCb")

                            time.sleep(8)

                            for r in related_images:
                                src2 = r.get_attribute("src")
                                if "http" in src2:
                                    if "jpg" in src2 or "jpeg" in src2 or "png" in src2:
                                        result.append([str(i) + "-" + str(j), src2])
                                        print(result)

            print("----")

            try:
                result_df = pd.DataFrame(result, columns=['index', 'img_url', 'related_images'])
                result_df.to_csv(file_name+".csv", mode='a', header=False, index=False)
            except IndexError:
                pass

            time.sleep(5)

        except ElementNotVisibleException:
            try:
                result_df = pd.DataFrame(result, columns=['index', 'img_url', 'related_images'])
                result_df.to_csv(file_name+".csv", mode='a', header=False, index=False)
            except IndexError:
                pass
        except ElementNotInteractableException:
            try:
                result_df = pd.DataFrame(result, columns=['index', 'img_url', 'related_images'])
                result_df.to_csv(file_name+".csv", mode='a', header=False, index=False)
            except IndexError:
                pass
        except ElementClickInterceptedException:
            try:
                result_df = pd.DataFrame(result, columns=['index', 'img_url', 'related_images'])
                result_df.to_csv(file_name+".csv", mode='a', header=False, index=False)
            except IndexError:
                pass
        except StaleElementReferenceException:
            try:
                result_df = pd.DataFrame(result, columns=['index', 'img_url', 'related_images'])
                result_df.to_csv(file_name+".csv", mode='a', header=False, index=False)
            except IndexError:
                pass
        except IndexError:
            pass

    driver.quit()
    driver.close()


def download_images_from_urls():

    urls = pd.read_csv("./train/stringing.csv")

    for i in range(0, len(urls)):
        print(urls.iloc[i]['url'])

        try:
            response = requests.get(urls.iloc[i]['url'])

            if urls.iloc[i]['valid'] == 1:
                file = open("./train/stringing/valid/" + str(urls.iloc[i]['index']) + ".jpg", "wb")
            else:
                file = open("./train/stringing/invalid/" + str(urls.iloc[i]['index']) + ".jpg", "wb")
            file.write(response.content)
            file.close()
        except SSLError:
            pass
        except MaxRetryError:
            pass

def get_image_title():

    urls = pd.read_csv("./result.csv")

    title = []

    for i in range(0, len(urls)):

        url_parts = re.split("/|\?", urls.iloc[i][1])

        for part in url_parts:
            if "jpg" in part:
                title.append(part.split(".jpg")[0])
                break
            elif "jpeg" in part:
                title.append(part.split(".jpeg")[0])
                break
            elif "png" in part:
                title.append(part.split(".png")[0])
                break

    urls['title'] = title

    urls.to_csv("./result_title_added.csv", index=False)


def get_valid_images():

    images = pd.read_csv("./image-urls-duplicates-marked.csv")

    valid = []

    for i in range(len(images)):
        # valid images-stringing-support
        if images.iloc[i]['valid'] != 1:
            valid.append(images.iloc[i]['index'])

            try:
                response = requests.get(images.iloc[i]['image_url'])

                image_format = images.iloc[i]['image_url'].split(".")[-1]

                if "?" in image_format:
                    image_format = image_format.split("?")[0]

                file = open("./valid_images/" + str(images.iloc[i]['index']) + "." + image_format, "wb")
                file.write(response.content)
                file.close()
            except SSLError:
                pass
            except MaxRetryError:
                pass

    #os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")


if __name__ == "__main__":
    download_images_from_urls()

