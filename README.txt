ARM Lab 05 - Robot poszukujący
================================

Robot TurtleBot3 Waffle przeszukuje mieszkanie w symulacji i znajduje
5 kostek ArUco. Skrypt nakłada siatkę punktów co 1,5 m, jeździ po nich,
obraca się w każdym punkcie i wypisuje ID oraz pozycje znalezionych kostek.


JAK URUCHOMIĆ
-------------

1. Zbudowanie obrazu Dockera (tylko za pierwszym razem):

   cd ~/arm/lab05
   ./arm_05_build.sh

2. Uruchomienie kontenera:

   xhost +local:root
   ./arm_05_run_gpu_nvidia.sh

3. Skopiowanie skryptu do kontenera (w nowym terminalu, na hoście):

   cd ~/arm/lab05
   docker cp lab05_siatka.py ARM_05:/arm_ws/lab05_siatka.py

4. Terminal 1 - symulacja:

   docker exec -it ARM_05 bash
   cd /arm_ws
   source install/setup.bash
   ros2 launch arm05_sim world.launch.py

5. Terminal 2 - RViz:

   docker exec -it ARM_05 bash
   cd /arm_ws
   source install/setup.bash
   ros2 launch nav2_bringup rviz_launch.py use_sim_time:=true

6. Terminal 3 - skrypt (zaraz po starcie symulacji, robot musi stać w (0, 0)):

   docker exec -it ARM_05 bash
   cd /arm_ws
   source install/setup.bash
   python3 lab05_siatka.py


PODGLĄD W RVIZ
--------------

Add -> By topic -> Marker dla dwóch topiców:

   /punkty_siatki   - zielone punkty siatki
   /kostki          - niebieskie kostki z napisem ID

Fixed Frame musi być ustawione na: map


SPRAWDZENIE WYNIKU
------------------

Prawdziwe pozycje kostek (w kontenerze):

   cat /root/.ros/arm05_sim/logs/latest.txt


PROBLEMY
--------

- Robot źle się lokalizuje: zrestartuj symulację albo ustaw pozycję
  w RViz przyciskiem "2D Pose Estimate".
- Robot ma problem przy ścianach: zmniejsz MAX_KOSZT w lab05_siatka.py.
- Za mało punktów siatki: zwieksz MAX_KOSZT albo zmniejsz KROK.
