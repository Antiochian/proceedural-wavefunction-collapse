# wavefunction-collapse

A proceedural image generator using a probability wavefunction collapse algorithm. Every 3x3 pixel grid exists as a superposition of all possible allowed 3x3 pixel grids according to an internal set of rules, and as each pixel updates it affects the superposition of the rest of the image by disallowing/allowing various of the possible configurations.


![Junctions](simple_snake_crossover.gif =100x) ![No junctions](simple_snake.gif)

If you look closely at the boundaries between the collapsed and uncollapsed regions of the image, you can see the various superpositions per-pixel as varying shades of gray (brighter = more likely to be white).

![Superposition closeup](superposition.png)

At the moment, all there is is a hardcoded proof of concept using simple black+white lines.

In the simplesnake.py program, the algorithm has been hardcoded to avoid there ever being any dead ends in the white path but eventually I want to experiment with letting the algorithm try and decide its own rules based on a training image instead. I think in theory the concept should be very similar to a Markov chain, but a 2-dimensional one, if that makes sense.
