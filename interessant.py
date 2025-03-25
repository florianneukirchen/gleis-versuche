import os 

basedir = "/media/riannek/minimax/gleis"
run24 = os.path.join(basedir, "2024-08-13/01/run24/01")
run14 = os.path.join(basedir, "2024-08-14/01/run14/01")

steigung = ["steigung1.laz", "steigung2.laz", "steigung3.laz", "steigung4.laz"]

interessant = {
    "Einfach": "4473900_5335875.copc.laz",
    "Diagonal": "4480300_5356625.copc.laz",
    "Weiche": "4473725_5336900.copc.laz",
    "Ende": "4473775_5336700.copc.laz",
    "Bahnübergang": "4473950_5335400.copc.laz",
    "Bahnübergang 2": "4474325_5332875.copc.laz",
    "Zug run 14 A" : "4480675_5356875.copc.laz",
    "Zug run 14 B": "4480600_5356850.copc.laz",
    "Zug run 24 A": "4478575_5349350.copc.laz",
    "Zug run 24 B": "4478575_5349425.copc.laz",
    "Gebäude": "4473750_5336925.copc.laz",
    "Gleis hohe Intensität": "4482100_5357075.copc.laz",
    "Gleis hohe Intensität 2": "4482150_5357075.copc.laz",
    "Feldweg": "4479700_5355900.copc.laz",
    "Feld": "4479350_5354800.copc.laz",
    "Weiche A": "4479025_5352900.copc.laz",
    "Weiche B": "4479025_5352925.copc.laz",
    "Zaun": "4478775_5351900.copc.laz",
    "Straße": "4478575_5349925.copc.laz",
    "Kante in Graben": "4478600_5348650.copc.laz",
    "Unterirdischer Bhf":"4476275_5343125.copc.laz",
    "Fangschiene Tunnel": "4474500_5340125.copc.laz",
    "Fangschiene Tunnel 2": "4473750_5338925.copc.laz",
    "Gleis weit abseits": "4473625_5337975.copc.laz",
    "Komische Linie": "4473725_5337650.copc.laz",
    "Kabeltöpfe": "4473725_5337575.copc.laz",
    "Güterzug": "4473700_5337500.copc.laz",
    "Güterzug Ende": "4473700_5337225.copc.laz",
    "Betondeckel": "4473700_5337175.copc.laz",
    "Bahnsteig": "4473775_5336600.copc.laz",
    "Bahnsteig Ende": "4473750_5336775.copc.laz",
    "Ding neben Gleis": "4473825_5336225.copc.laz",
    "Wände": "4474075_5333750.copc.laz",
    "Viele Gleise": "4474750_5332150.copc.laz",
    "Anfang Weiche": "4481275_5357000.copc.laz",
    "OLA gleiche Höhe wie Gleis": "4473675_5337950.copc.laz",
    "4 Gleise": "4481800_5357050.copc.laz",
    "Y": "4481250_5356975.copc.laz", 
    "Weiche C": "4473850_5336225.copc.laz",
    "Weiche D": "4473850_5336150.copc.laz",
    "Für template": "4475250_5340950.copc.laz",
    "Rand": "4474075_5333550.copc.laz",
    "Extrem viele Punkte run24": "4474800_5332150.copc.laz",
    "Viele Gleise 2": "4480875_5356950.copc.laz",
    "Kreuzung linker Teil": "4480900_5356950.copc.laz",
    "Kreuzung rechter Teil": "4480925_5356950.copc.laz",
}



"""
0 Einfach
1 Diagonal
2 Weiche
3 Ende
4 Bahnübergang
5 Bahnübergang 2
6 Zug run 14 A
7 Zug run 14 B
8 Zug run 24 A
9 Zug run 24 B
10 Gebäude
11 Gleis hohe Intensität
12 Gleis hohe Intensität 2
13 Feldweg
14 Feld
15 Weiche A
16 Weiche B
17 Zaun
18 Straße
19 Kante in Graben
20 Unterirdischer Bhf
21 Fangschiene Tunnel
22 Fangschiene Tunnel 2
23 Gleis weit abseits
24 Komische Linie
25 Kabeltöpfe
26 Güterzug
27 Güterzug Ende
28 Betondeckel
29 Bahnsteig
30 Bahnsteig Ende
31 Ding neben Gleis
32 Wände
33 Viele Gleise
34 Anfang Weiche
35 OLA gleiche Höhe wie Gleis
36 4 Gleise
37 Y
38 Weiche C
39 Weiche D
40 Für template
41 Rand
42 Extrem viele Punkte run24
43 Auch viele Gleise

"""
