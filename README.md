#BERTextFilter

Ez a projekt egy szövegfeldolgozó pipeline-t biztosít, amely magában foglalja a szövegfájlok nyelvhelyességi vizsgálata alápján a szűrést, a duplikált mondatok kezelését és opcionálisan egy Hugging Face adatcsomag létrehozását.

##Telepítés

    Klónozd a repozitóriumot:
        git clone https://git.nlp.nytud.hu/madaraszg/BERTextFilter.git
        
    Telepítsd a szükséges csomagokat:
    ```bash
    pip install -r requirements.txt
    ```
    
##Használat

A fő script futtatásához:

    ```bash
    python3 bertextfilter.py
    ```

##Fő Script

A fő script az alábbi lépéseket hajtja végre:
    Bekéri a felhasználótól a használni kívánt GPU ID-át.
    Bekéri a felhasználótól a bemeneti mappa elérési útját és a kimeneti mappa elérési útját.
    Ellenőrzi a bemeneti mappa létezését.
    Létrehozza a kimeneti mappát, ha nem létezik.
    Feldolgozza a bemeneti mappában lévő szövegfájlokat a text_cleaner.py-ban található process_all_files függvény segítségével.
    Megkeresi és kilistázza a duplikált mondatokat a feldolgozott fájlokban a deduplicate.py-ban található main függvény segítségével.
    Megkérdezi a felhasználót, hogy szeretné-e törölni a duplikált mondatokat.
    Megkérdezi a felhasználót, hogy szeretne-e Hugging Face adatcsomagot létrehozni:
        Ha igen, bekéri az adatcsomag nevét és létrehozza az adatcsomagot a dataset_creation.py-ban található main függvény segítségével.

