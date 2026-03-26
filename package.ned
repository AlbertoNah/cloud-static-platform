package inet;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.WirelessHost;
import inet.physicallayer.wireless.ieee80211.packetlevel.Ieee80211ScalarRadioMedium;

network TwoNodeNetwork
{
    parameters:
        @display("bgb=500,400");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=200,50");
        }
        radioMedium: Ieee80211ScalarRadioMedium {
            @display("p=300,50");
        }
        baseStation: WirelessHost {
            @display("p=200,200");
        }
        node1: WirelessHost {
            @display("p=350,200");
        }
}

network MobilityNetwork
{
    parameters:
        @display("bgb=600,500");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=200,50");
        }
        radioMedium: Ieee80211ScalarRadioMedium {
            @display("p=300,50");
        }
        baseStation: WirelessHost {
            @display("p=300,250");
        }
        node[5]: WirelessHost {
            @display("p=100,100");
        }
}

network SmallIoTNetwork
{
    parameters:
        @display("bgb=600,500");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=200,50");
        }
        radioMedium: Ieee80211ScalarRadioMedium {
            @display("p=300,50");
        }
        baseStation: WirelessHost {
            @display("p=300,250");
        }
        node[7]: WirelessHost {
            @display("p=100,100");
        }
}
