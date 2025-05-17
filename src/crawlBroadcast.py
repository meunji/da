from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime
import pandas as pd
import os

def get_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창 안 띄움
    chrome_options.add_argument("--no-sandbox")  # 샌드박스 끔
    chrome_options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 방지
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")

    chrome_options.binary_location = "/usr/bin/chromium"

    return webdriver.Chrome(
        service=Service("/usr/lib/chromium-browser/chromedriver"),
        options=chrome_options
    )


def get_date(base_year, base_month, base_day, d) -> datetime:
    base_date = datetime(base_year, base_month, base_day)

    for i in range(-2, 7):
        check_date = base_date + timedelta(days=i)
        if check_date.day == d:
            return check_date.strftime('%Y-%m-%d')


def crawl_broadcast_info():
    today = datetime.today()

    url = 'https://live.ecomm-data.com/schedule/hs'
    # browser = webdriver.Chrome() 
    browser = get_chrome_driver()
    browser.get(url)
    
    # 1. 날짜별로 조회.
    search_date_list = browser.find_elements('css selector', "ul > li > div.schedule_cell__0ZCso")
    
    for i in range(len(search_date_list)):
        search_date_list = browser.find_elements('css selector', "ul > li > div.schedule_cell__0ZCso")
        search_button = search_date_list[i]
        
        search_button.click()
        time.sleep(2)
    
        for j in range(5):
            body = browser.find_element('css selector', 'body')
            body.send_keys(Keys.HOME)
            time.sleep(1)
        
        # 크롤링
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        day = search_button.text.split()[0]
    
        search_date = get_date(today.year, today.month, today.day, int(day))
        # search_date = f'{year}-{month}-{day}'
        print(search_date)
    
    
        results = []
    
        # 날짜별 전체 편성 가져오기
        day_div_list = soup.select('div.schedule_container__Bp3qt') 
        
        # day_div_list[1].select('div.Table_container__cUG9N')[0].select('table.Table_table___jpMW')[0].select('tr')[2].select('td')[1].select('span')[0].text
        
        for day in day_div_list[1:]:
            time_div_list = day.select('div.Table_container__cUG9N')[0].select('table.Table_table___jpMW')[0].select('tr') 
        
            for broad in time_div_list:
                td_list = broad.select('td')
                
                if len(td_list) >= 4 and td_list[1].select('span'):
                    try:
                        date_ymd = td_list[1].select('span')[0].text
                        time_hm = td_list[1].select('span')[1].text
                        broad_info = td_list[2].text.strip()
                        company_info = td_list[2].select('span.TableSchedule_adWrap__V_CMK')[0].text.strip() if td_list[2].select('span.TableSchedule_adWrap__V_CMK') else ''
                        category = td_list[3].text.strip()
        
                        # print(date_ymd, time_hm, broad_info, company_info, category)
                        
                        data = [date_ymd, time_hm, broad_info, company_info, category]
                        results.append(data)
        
                    except IndexError as e:
                        print("IndexError 발생한 row:", broad)
                        print("에러 메시지:", e)
    
        # csv로 데이터 저장        
        columns = ['date', 'time', 'broad_info', 'company_info', 'category']
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(f'./broad_info/broad_{search_date}.csv', index=False)

def save_file_broadcast_info():
    total_file = pd.DataFrame()
    
    for filename in os.listdir('./broad_info'):
        if filename[-3: ] == 'csv':
            day_file = pd.read_csv(f'./broad_info/{filename}')
            total_file = pd.concat([total_file, day_file], ignore_index=True)

    total_file.to_csv('./file/broad_info.csv', index=False, encoding='utf-8-sig')

def crawl_main():
    crawl_broadcast_info()
    save_file_broadcast_info()

# main() 함수 실행
if __name__ == "__main__":
    crawl_main()
