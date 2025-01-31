import requests
from bs4 import BeautifulSoup
import csv

start_url = "https://en.wikipedia.org/wiki/Special:AllPages?from=xxxxxxxxx"

response = requests.get(start_url)
if response.status_code != 200:
    print(f" HTTP Error: {response.status_code}")
    exit()

soup = BeautifulSoup(response.content, "html.parser")
allpages_div = soup.find("div", class_="mw-allpages-body")
if allpages_div is None:
    print("Error: 'mw-allpages-body' has been not found!")
    exit()

links = allpages_div.find_all("a", href=True)

csv_filename = "x.csv" 
with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Article Name", "Content"])  

    def process_article(link_text, link_href):
        full_url = f"https://en.wikipedia.org{link_href}"
        print(f"\nProcessing: {link_text} -> {full_url}")
        
        response = requests.get(full_url)
        if response.status_code != 200:
            print(f"Error: {link_text} has been not download ! HTTP Error: {response.status_code}")
            return

        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find("div", class_="mw-content-ltr mw-parser-output")
        if content_div is None:
            print(f"Error: {link_text} could not find the content!")
            return

        article_data = []
        for element in content_div.find_all(["h2", "h3", "p"]):
            if element.name in ["h2", "h3"]:
                article_data.append(f"\n== {element.get_text(strip=True)} ==")
            elif element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    article_data.append(text)

        full_article = "\n".join(article_data)

        writer.writerow([link_text, full_article])
        print(f"{link_text} has been saved.")

    for i, link in enumerate(links, start=1):
        link_text = link.get_text(strip=True)
        link_href = link['href']
        print(f"\n#{i} Article Processing...")
        process_article(link_text, link_href)