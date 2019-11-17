        plt.figure()     
        nx.draw(G, pos, edge_color="Grey", cmap=plt.get_cmap('RdYlGn_r'), node_color=[G.node[x]['infected'] for x in G.nodes()], vmin=0, vmax=1.0)
        adjust the plot limits
        cut = 1.14
        xmax= cut*max(xx for xx,yy in pos.values())
        ymax= cut*max(yy for xx,yy in pos.values())   
        plt.xlim([-xmax, xmax])
        plt.ylim([-ymax, ymax])
        plt.savefig('imgs/%04d.png'%len(t), dpi = 300)    
        plt.close('all')
        