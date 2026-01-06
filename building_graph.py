import heapq
from collections import defaultdict

class BuildingGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.blocked_edges = set()

    def add_room_connection(self, a, b, distance=1):
        if (a, b) in self.blocked_edges or (b, a) in self.blocked_edges:
            return
        self.graph[a].append((b, distance))
        self.graph[b].append((a, distance))

    def block_connection(self, a, b):
        self.blocked_edges.add((a, b))
        self.blocked_edges.add((b, a))

    def dijkstra(self, start, end):
        pq = [(0, start, [])]
        visited = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == end:
                return cost, path

            for neighbor, w in self.graph[node]:
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + w, neighbor, path))

        return None, []


def create_building():
    building = BuildingGraph()

    building.block_connection("3.006", "3.007")
    building.block_connection("3.104", "3.106")
    building.block_connection("3.2.04", "3.2.06")


    # Keller (K) –

    # Erdgeschoss (0)
        # Haus1
    building.add_room_connection("Treppenhaus_1", "Bibliothek")
    building.add_room_connection("Treppenhaus_1", "Mensa")
    building.add_room_connection("Treppenhaus_1", "1.001")
    building.add_room_connection("Treppenhaus_1", "1.202")
    building.add_room_connection("1.202", "1.201")
    building.add_room_connection("1.001", "1.002")


        # Haus2
    building.add_room_connection("3.005", "3.006")
    building.add_room_connection("3.005","2.020")
    building.add_room_connection("2.020", "2.019")
    building.add_room_connection("2.019", "2.018")
    building.add_room_connection("2.018", "Treppenhaus_2_Fahrstuhl")
    building.add_room_connection("Treppenhaus_2_Fahrstuhl", "2.014")
    building.add_room_connection("2.014", "2.013")
    building.add_room_connection("2.013", "2.012")
    building.add_room_connection("2.012", "2.011")
    building.add_room_connection("2.011", "Treppenhaus_2_Foyer")
    building.add_room_connection("2.014","2.015")
    building.add_room_connection("2.015","2.016")
    building.add_room_connection("2.016","2.017")
    building.add_room_connection("Treppenhaus_2_Foyer","2.010")
    building.add_room_connection("2.010","2.009")
    building.add_room_connection("2.009","2.008")
    building.add_room_connection("2.008","2.007")
    building.add_room_connection("Treppenhaus_2_Foyer","2.001")
    building.add_room_connection("2.001","2.WC-Damen-EG")
    building.add_room_connection("2.WC-Damen-EG","2.WC-Herren-EG")
    building.add_room_connection("2.WC-Herren-EG","2.004")
    building.add_room_connection("2.004","2.005")

        #Haus3
    building.add_room_connection("3.007", "3.004")
    building.add_room_connection("3.004", "3.007")
    building.add_room_connection("3.007", "3.006")
    building.add_room_connection("3.006", "3.009")
    building.add_room_connection("3.009", "3.010")
    building.add_room_connection("3.010", "Treppenhaus_3")
    building.add_room_connection("Treppenhaus_3", "3.WC-Herren-EG")
    building.add_room_connection("3.WC-Herren-EG", "3.XX-O1")
    building.add_room_connection("3.XX-O1", "3.XX-02")
    building.add_room_connection("3.XX-02", "3.003")
    building.add_room_connection("Treppenhaus_3", "3.011")
    building.add_room_connection("Treppenhaus_3", "3.012")
    building.add_room_connection("Treppenhaus_3", "3.015")
    building.add_room_connection("3.015", "3.014")
    building.add_room_connection("3.014", "3.013")


    # 1 OG
    #Haus 2
    building.add_room_connection("3.105", "3.104")
    building.add_room_connection("3.104", "2.119")
    building.add_room_connection("2.119", "2.118")
    building.add_room_connection("2.118", "Treppenhaus_2_Fahrstuhl")
    building.add_room_connection("Treppenhaus_2_Fahrstuhl", "2.115")
    building.add_room_connection("2.115", "2.114")
    building.add_room_connection("2.115", "2.113")
    building.add_room_connection("2.113", "2.112")
    building.add_room_connection("2.112", "2.111")
    building.add_room_connection("2.111", "Treppenhaus_2_Foyer")
    building.add_room_connection("2.115","2.116")
    building.add_room_connection("2.116","2.117")
    building.add_room_connection("2.117","Treppenhaus_2_Foyer")
    building.add_room_connection("Treppenhaus_2_Foyer","2.110")
    building.add_room_connection("2.110","2.109")
    building.add_room_connection("2.109","2.108")
    building.add_room_connection("2.108","2.107")
    building.add_room_connection("Treppenhaus_2_Foyer","2.101")
    building.add_room_connection("2.101","2.WC-Damen-1OG")
    building.add_room_connection("2.WC-Damen-1OG","2.WC-Herren-1OG")
    building.add_room_connection("2.WC-Herren-1OG","2.104")
    building.add_room_connection("2.104","2.105")
    building.add_room_connection("2.105","2.106")

    #Haus3
    building.add_room_connection("3.106", "3.107")
    building.add_room_connection("3.107", "3.108")
    building.add_room_connection("3.108", "3.109")
    building.add_room_connection("3.109", "3.110")
    building.add_room_connection("3.107","3.103")
    building.add_room_connection("3.103","3.102")
    building.add_room_connection("3.102","3.WC-Damen_1OG")
    building.add_room_connection("3.WC-Damen_1OG","Treppenhaus_3")
    building.add_room_connection("Treppenhaus_3","3.110")
    building.add_room_connection("Treppenhaus_3","3.111")
    building.add_room_connection("Treppenhaus_3","3.112")
    building.add_room_connection("Treppenhaus_3","3.113")
    building.add_room_connection("Treppenhaus_3","3.114")
    building.add_room_connection("3.114","3.115")
    building.add_room_connection("3.115","3.117")
    building.add_room_connection("3.117","3.118")
    building.add_room_connection("3.114","3.119")



    # 2. OG
    building.add_room_connection("3.204","3.104")
    building.add_room_connection("3.204","3.205")
    building.add_room_connection("2.234","Treppenhaus_2_Fahrstuhl")
    building.add_room_connection("Treppenhaus_2_Fahrstuhl","2.228")
    building.add_room_connection("2.228","2.227")
    building.add_room_connection("2.227","2.226")
    building.add_room_connection("2.228","2.225")
    building.add_room_connection("2.225","2.224")
    building.add_room_connection("2.224","2.223")
    building.add_room_connection("2.223","2.222")
    building.add_room_connection("2.222","2.221")
    building.add_room_connection("2.221","Treppenhaus_2_Foyer")
    building.add_room_connection("2.228","2.229")
    building.add_room_connection("2.229","2.230")
    building.add_room_connection("2.230","2.231")
    building.add_room_connection("2.231","2.232")
    building.add_room_connection("2.232","2.233")
    building.add_room_connection("2.233","Treppenhaus_2_Foyer")
    building.add_room_connection("Treppenhaus_2_Foyer","2.220")
    building.add_room_connection("Treppenhaus_2_Foyer","2.219")
    building.add_room_connection("Treppenhaus_2_Foyer","2.218")
    building.add_room_connection("2.218","2.217")
    building.add_room_connection("2.217","2.216")
    building.add_room_connection("2.216","2.215")
    building.add_room_connection("2.215","2.214")
    building.add_room_connection("2.214","2.213")
    building.add_room_connection("2.213","2.212")
    building.add_room_connection("2.212","2.211")
    building.add_room_connection("2.211","2.210")
    building.add_room_connection("2.210","2.209")
    building.add_room_connection("Treppenhaus_2_Foyer","2.201")
    building.add_room_connection("2.201","2.WC-Damen-2OG")
    building.add_room_connection("2.WC-Damen-2OG","2.WC-Herren-2OG")
    building.add_room_connection("2.WC-Herren-2OG","2.204")
    building.add_room_connection("2.204","2.205")
    building.add_room_connection("2.205","2.206")
    building.add_room_connection("2.206","2.207")
    building.add_room_connection("2.207","2.208")
    building.add_room_connection("2.208","2.209")

    #Haus3
    building.add_room_connection("3.206","3.207")
    building.add_room_connection("3.207","3.208")
    building.add_room_connection("3.208","3.209")
    building.add_room_connection("3.209","3.210")
    building.add_room_connection("3.210","3.211")
    building.add_room_connection("3.211","3.212")
    building.add_room_connection("3.212","Treppenhaus_3")
    building.add_room_connection("Treppenhaus_3","3.201")
    building.add_room_connection("3.201","3.202")
    building.add_room_connection("3.202","3.203")
    building.add_room_connection("Treppenhaus_3","3.213")
    building.add_room_connection("Treppenhaus_3","3.214")
    building.add_room_connection("Treppenhaus_3","3.215")
    building.add_room_connection("Treppenhaus_3","3.216")
    building.add_room_connection("3.216","3.218")
    building.add_room_connection("3.218","3.220")


    # Keller (K) –

    #Ausgänge
    building.add_room_connection("Treppenhaus_2_Foyer", "Innenhof")
    building.add_room_connection("Innenhof", "Treppenhaus_3")

    return building