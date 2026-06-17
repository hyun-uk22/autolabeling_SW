import os
import urllib.request

def download_samples():
    os.makedirs("data/raw", exist_ok=True)
    
    # 10 diverse sample images for a solid mini-experiment
    samples = [
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg", "01_dog_bike_car.jpg"),
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg", "02_eagle.jpg"),
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg", "03_person_horse.jpg"),
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg", "04_horses.jpg"),
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg", "05_people_kites.jpg"),
        ("https://ultralytics.com/images/zidane.jpg", "06_zidane.jpg"),
        ("https://ultralytics.com/images/bus.jpg", "07_bus.jpg"),
        ("https://farm3.staticflickr.com/2436/3811654877_ab3d2e92c2_z.jpg", "08_cat.jpg"), # COCO sample
        ("https://farm4.staticflickr.com/3336/3277713495_25f0e95738_z.jpg", "09_pizza.jpg"), # COCO sample
        ("https://farm1.staticflickr.com/151/421448887_fde752ecac_z.jpg", "10_baseball.jpg") # COCO sample
    ]
    
    print("Downloading 10 sample images for experiment...")
    for url, name in samples:
        path = os.path.join("data/raw", name)
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
                print(f" [+] Downloaded {name}")
            except Exception as e:
                print(f" [-] Failed to download {name}: {e}")
        else:
            print(f" [=] {name} already exists.")
            
    print("Download complete. Images are in 'data/raw/'.")

if __name__ == "__main__":
    download_samples()
