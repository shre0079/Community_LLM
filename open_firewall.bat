@echo off
echo Opening OLMoE firewall ports...
netsh advfirewall firewall add rule name="OLMoE-Coord"   dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="OLMoE-Worker"  dir=in action=allow protocol=TCP localport=8001
netsh advfirewall firewall add rule name="OLMoE-ZMQ-In"  dir=in action=allow protocol=TCP localport=5555
netsh advfirewall firewall add rule name="OLMoE-ZMQ-Ret" dir=in action=allow protocol=TCP localport=5556
echo Done.
pause