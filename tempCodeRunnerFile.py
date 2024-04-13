plt.plot(self.start.x, self.start.y, "xr")
		plt.plot(self.end.x, self.end.y, "xr")
		plt.axis([-15, 15, -15, 15])
		plt.grid(True)
		plt.savefig('image%04d'%image_counter)
		image_counter += 1
		plt.pa