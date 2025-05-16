from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By 
import time
import os

def get_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창 안 띄움
    chrome_options.add_argument("--no-sandbox")  # 샌드박스 끔
    chrome_options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 방지
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")

    chrome_options.binary_location = "/usr/bin/chromium-browser"

    return webdriver.Chrome(
        service=Service("/usr/bin/chromedriver"),
        options=chrome_options
    )

def remote_bank_order(driver):
    # 바로주문
    webdriver.ActionChains(driver).send_keys(Keys.ARROW_DOWN).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4)

    # 상품 옵션 선택
    # webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    # time.sleep(4) 

    # 주문방법 선택 > 리모컨 주문
    webdriver.ActionChains(driver).send_keys(Keys.ARROW_RIGHT).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4) 

    # 상품 속성, 수량 선택
    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4) 

    webdriver.ActionChains(driver).send_keys(Keys.ARROW_DOWN).perform()
    time.sleep(4) 

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4) 

    # 결제수단 선택 > 무통장 입금
    webdriver.ActionChains(driver).send_keys(Keys.ARROW_RIGHT).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4)

    # 은행종류 선택
    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4)

    # 결제
    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(10)

    # 확인
    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(10)

def cancel_order(driver):
    # 주문 취소 > 내정보 > 주문배송현황
    webdriver.ActionChains(driver).send_keys(Keys.ARROW_DOWN).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ARROW_DOWN).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(2)

    webdriver.ActionChains(driver).send_keys(Keys.ARROW_LEFT).perform()
    time.sleep(6)

    # webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    # time.sleep(7)

    webdriver.ActionChains(driver).send_keys(Keys.ARROW_DOWN).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(5)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(5)

    webdriver.ActionChains(driver).send_keys(Keys.ARROW_LEFT).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4)

    # 주문 취소 팝업 확인
    webdriver.ActionChains(driver).send_keys(Keys.ARROW_LEFT).perform()
    time.sleep(4)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(10)

    # 취소 완료 후 확인
    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(10)

    webdriver.ActionChains(driver).send_keys(Keys.ENTER).perform()
    time.sleep(4)




def order_main():
    svcSeq = '51'
    url = 'http://123.111.139.135:8180/main_ani.jsp?svcSeq=' + svcSeq + '&graphicResolution=2#platformNameId=8&clientType=1&deviceType=1&videoResolution=2&deviceId=LC_PC_TEST&soId=LC_PC&model=NONE&mac=LC_PC_MAC&bridged=false'
    
    # driver = webdriver.Chrome()
    driver = get_chrome_driver()
    driver.get(url)
    driver.maximize_window()

    element = driver.find_element(By.CSS_SELECTOR, "#goodsDetail .leftArea .btm .btnArea .btns")

    if element.get_attribute('innerHTML') == "상담하기":
        print('무형상품 주문')
    else :
        # 무통장 주문
        remote_bank_order(driver)
        # 주문 취소
        cancel_order(driver)

# main() 함수 실행
if __name__ == "__main__":
    order_main()