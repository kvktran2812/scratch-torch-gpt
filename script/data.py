from bs4 import BeautifulSoup
import selenium
import requests


# constants
thi_vien_url = "https://www.thivien.net" 
main_url = "https://www.thivien.net/Nguy%E1%BB%85n-Du/Truy%E1%BB%87n-Ki%E1%BB%81u/group-uAY7gIaARbh2b4DCVporPQ"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}


def get_poem_group_list(url):
    # function variables
    data = [] # data variable contains all poem info

    # fetch url and web scraping to collect data
    response = requests.get(main_url, headers=headers)
    soup = BeautifulSoup(response.content, "lxml")
    poem_group_list = soup.find("div", {"class": "poem-group-list"}) # find div with class "poem-group-list" that contains all poems
    poem_group_list = poem_group_list.find_all("li") # get only the <li> element

    # for each element in the poem_group_list, get the href and the text
    # store the information in a dict and add to the data list
    for poem in poem_group_list:
        link = poem.find("a")
        data.append({
            "href": thi_vien_url + link["href"],
            "text": link.text,
        })

    return data