from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Set up Chrome webdriver
driver = webdriver.Chrome()
driver.maximize_window()

# Open the webpage
driver.get("https://genius.com/artists/Engelbert-humperdinck/songs")

time.sleep(3)

popularity_button = driver.find_element(By.XPATH, "//*[contains(text(),'Popularity')]")
popularity_button.click()

time.sleep(2)

# sorting song titles alphabetically
popularity_button = driver.find_element(By.XPATH, "//*[contains(text(),'A-Z')]")
popularity_button.click()

time.sleep(2)

# Scroll down until the last song "You’ve Really Got a Hold on Me" appears
while True:
    # Scroll down to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Wait for a moment
    time.sleep(3)  
    
    # Scroll slightly up
    driver.execute_script("window.scrollBy(0, -100);")
    
    # Check if the last song appears on the page
    try:
        song_element = driver.find_element(By.XPATH, "//h3[contains(text(), 'You’ve Really Got a Hold on Me')]")
        break
    except:
        pass  

# processing current HTML contant using beautiful soup
current_html = driver.page_source
soup = BeautifulSoup(current_html, "html.parser")

elements = soup.find_all(class_="ListItem__Link-sc-122yj9e-1")

song_links = [] 
song_titles = []
all_lyrics = []

# extracting song links:
for tag in elements:
    href_value = tag['href']
    song_links.append(href_value)
    # print(href_value)
    
# each song need separate html request:
for link in song_links:
    song_response = requests.get(link)
    song_soup = BeautifulSoup(song_response.text, "html.parser")
    
    # scraping song title:
    title_element = song_soup.find("span", class_="SongHeaderdesktop__HiddenMask-sc-1effuo1-11 iMpFIj")
    try:
        song_title = title_element.text.strip()
        song_titles.append(song_title)
    except:
        song_titles.append("na")
    
    # scraping song lyrics:
    lyrics_div = song_soup.find("div", {"data-lyrics-container": True})
    
    # if soup.find() does not find any element matching the specified criteria, 
    # it will return None. Type of such element will be NoneType and can cause error.
    # Some songs can have no lyrics available 
    try:
        lyrics = lyrics_div.get_text(separator="\n")
        all_lyrics.append(lyrics.strip())
    except:
        all_lyrics.append("na")

df = pd.DataFrame({'title': song_titles, 'lyrics': all_lyrics})

df['lyrics'] = df['lyrics'].str.replace("\n", " ")

df.to_excel("engelbert_songs.xlsx", index=False) # mode='w'


# Close the browser window
driver.quit()