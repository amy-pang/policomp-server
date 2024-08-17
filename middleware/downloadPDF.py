from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import os
import glob

# Set options for webdriver that allow pdfs to be downloaded automatically
options = webdriver.ChromeOptions()

options.add_experimental_option('prefs', {
    "download.default_directory": "/Users/amyyp/Desktop/dev/policomp/data",
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

url = 'https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/party-platforms-and-nominating-conventions-3'
driver.get(url)

# Prevents errors while accessing elements
driver.maximize_window()
driver.implicitly_wait(5)

# Clear PDFs currently stored
directory_path = '/Users/amyyp/Desktop/dev/policomp/data'
files = glob.glob(os.path.join(directory_path, '*'))

for file in files:
    try:
        os.remove(file)
        print(f'Successfully deleted: {file}')
    except Exception as e:
        print(f'Error deleting file {file}: {e}')

# Access top 2 pdfs of presidential candidate platforms, which should be that of the candidates of the most recent election
all_a_tags = driver.find_elements(By.XPATH, "//a")
count = 0

for a in all_a_tags:
    try:
        if a.get_attribute('href') and '.pdf' in a.get_attribute('href') and 'pdf' in a.text.lower():
            a.click()
            
            count += 1
            if count >= 2:
                break

            print(f'Successfully downloaded pdf {count}')
    except Exception as e:
        print(f'Error downloading pdf {count}: {e}')

time.sleep(10)
