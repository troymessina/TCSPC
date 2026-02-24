# TCSPC
This is a collection of analysis programs for fluorescence microscopy and time-correlated single photon counting (TCSPC) experiments. The Jupyter notebooks can be opened in Google's Colab. Simply go to a notebook (.ipynb file). Change the html address from github.com to githubtocolab.com or from github.com to colab.research.google.com.

## Directory Descriptions:
* IPG - an implementation of interior point gradient methods to globally fit TCSPC lifetime data over an experimental coordinate (concentration, incubation time, etc.)
  * "Global fitting without a global model: Regularization based on the continuity of the evolution of parameter distributions," Jason T. Giurleo and David S. Talaga, *J. Chem. Phys.* **128**, 114114 (2008); doi: 10.1063/1.2837293

* HMM (more coming soon)
   * "Direct Determination of Kinetic Rates from Single-Molecule Photon Arrival Trajectories Using Hidden Markov Models," Michael Andrec, Ronald M. Levy, and David S. Talaga, *J. Phys. Chem. A* **107**(38), 7454-7464 (2003); doi: 10.1021/jp035514+
   * "Hidden Markov model analysis of multichromophore photobleaching," Troy C. Messina, Hiyun Kim, Jason T. Giurleo, and David S. Talaga, *J. Phys. Chem. B* **110** 16366-16376 (2006); doi: 10.1021/jp063367k
   * "Gold Ion Beam Milled Gold Zero-Mode Waveguides," Troy C. Messina, Bernadeta R. Srijanto, Charles Patrick Collier, Ivan I. Kravchenko, and Christopher I. Richards, *Nanomaterials* **12**(10), 1755 (2022); doi:10.3390

* Multiexponential TCSPC Lifetimes
  * Fit lifetime data with an arbitrary number of exponential functions and a convolved instrument response function.
* TCSPC Anisotropy - single shot magic, vertical, horizontal angle fluorescence analysis. Coming soon: Experimental coordinate analysis such as analysis of ligand concentration dependence.
  * "Protein Free Energy Landscapes Remodeled by Ligand Binding," Troy C. Messina and David S. Talaga, *Biophysical Journal* **93** 579-585 (2007); doi: 10.1529/biophysj.107.103911
