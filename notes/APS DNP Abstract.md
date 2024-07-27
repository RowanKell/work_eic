# APS DNP Abstract

## EIC KLM

#### Brainstorm

Big picture

1. Muon pion separation
2. Show endpoint plots

Simulation work

1. Editing geometry
   1. Full detector simulations with optical photons turned off
   2. Single bar (test stand-like) simulations with optical photons turned on
2. Parameterization
   1. Photon yield as function of position, bar length, and energy deposited
   2. Photon timing drawn from distribution
      1. Photon emission timing
      2. Photon travel time timing
   3. Able to get realistic statistics without computing photon physics - big time save
3. Preparing geometry for optimization

#### Outline

1. We conducted research and development on a Iron/Scintillator Calorimeter for a second detector at the Electron-Ion Collider. The Electron-Ion Collider (EIC) will be the first polarized electron-proton collider and will investigate the internal structure and dynamics of the nucleons. Accurate and efficient simulations are key to studying physics at the EIC. The present study continued the implementation of the detector geometry in a suitable detector description software allowing use in physics simulations. To improve computation efficiency of the simulations, research and development went into a parameterization of the optical photon simulation. This parameterization allows for simulating the physics of the detector without simulating optical photon physics, producing an accurate readout based on studies of the full photon simulation. The simulation software will be integral to producing projections for the EIC second detector, and will aid in the future physics studies at the EIC when data taking begins.

## Affinity

Big picture

1. single pion results at CLAS12 of TMD affinity

#### Outline

1. This study applies a new tool for measuring the applicability of factorization theorems, referred to as affinity, to the kinematics of JLab fixed target electron-proton scattering experiments. Different factorization theorems are valid for different kinematic regions of semi-inclusive deep inelastic scattering experiments. Affinity can be used to estimate the proximity of a kinematic bin from a factorization theorem, and relies on partonic kinematics to produce a value. The original work which proposed the tool produced projections for affinity for different experiments including JLab, utilizing simple models to approximate partonic kinematics which are not observable. The present study investigates the affinity of specific kinematics at JLab using partonic kinematics produced by Monte-Carlo simulations of the experiment. We find that the simulated partonic kinematics differ from the models used in the original work, opening the door to extensions based on more realistic partonic descriptions. Furthermore, we project affinity values with similar trends but  <results go here but for now...>  overall higher values of affinity in the TMD region, suggesting higher likelihood of TMD factorization applying to the JLab kinematics. </ END RESULTS> We also suggest a model for extending affinity to di-hadron physics where multiple final state hadrons are observed, allowing for higher degrees of freedom <THIS IS LIKELY NOT STAYING IN>

# CEU Application

### Research essay

`Please explain your **original** individual contribution to the larger group effort. This statement is intended to **supplement** your more formal research abstract (submitted separately to the APS). It  gives you the opportunity to describe more specifically your individual  contribution to the research. This supplementary information is  important to the review process and should give details about your  specific research contributions. This helps the committee understand  what your poster will actually contain.` 

1. `Do NOT repost your abstract.`
2. `This statement should also stand alone and should not refer to your  abstract directly, i.e. "I completed everything outlined in the  abstract."`
3. `Submit at least one paragraph but no more than one page.`

 `Upload this document in .pdf format with your full name included at the top of the page.` 



**Notes**

1. **Maybe mention that the majority of the work was programming / producing software to analyze the data and produce calculations**
   1. My thought is that they may want to hear what exactly I actually did, and besides meeting with people, the main thing I did was write code to process data and make calculations and plots


The proposed poster will present the my findings when investigating the applicability of the TMD factorization theorem to the kinematics of deep inelastic scattering experiments taking place at Jefferson Lab (JLab) using the CLAS12 detector. Transverse momentum dependent (TMD) factorization is used to factorize the scattering reaction into a perturbative scattering cross section, a nonperturbative TMD parton distribution function, and a fragmentation function. A tool, referred to as affinity, was recently proposed to help identify where factorization theorems, such as TMD factorization, hold up in phase space. My research project focused on applying this tool to the kinematics of JLab deep inelastic scattering (DIS) experiments to investigate how TMD factorization applicability varied with different kinematic variables. The focus of my research, has been on the affinity projections for JLab data. Although the initial proposal for the tool provided projections for JLab and other experiments, these projections were made using estimates of the kinematics rather than actual data. My project extends the initial study by creating projections based on the monte-carlo simulation data created for CLAS12 at JLab. These projections include affinity values calculated for different kinematic bins for charged pion final states which are abundant at CLAS12. 

Producing affinity projections for JLab kinematics requires processing the MC simulation data to allow for affinity calculations. I began by selecting events via cuts that corresponded to semi-inclusive DIS (SIDIS) reactions; I also selected for events with at least one final state pion. Once I only had SIDIS events I began looking at the partonic momenta which is required to make affinity calculations directly. Although partonic variables function no different from hadronic variables, partonic variables must be treated intentionally as they are not observable and are dependent on the theoretical models used to create the simulation. As a first step, I met with the maintainer of the simulation software used to generate the MC simulation to ensure that I understood which values in the data referred to what particles in the reactions and how to map the data from the simulation output to kinematic variables. Having understood how the partonic data corresponds to the kinematic models, I began making calculations of affinity. Immediately my research group and I could view the results and see where in phase space TMD factorization should apply based on my work.

With the success in producing projections from the JLab MC I brought my findings to the researchers who first proposed the affinity tool to discuss how the results fit into each of our understandings of the theory behind the model. We found that my projections differed from the original projections when looking at the magnitude of the TMD affinity, yet we found similar trends in both the original findings and my new projections. After investigation, we found that using MC simulation data has a significant effect on the resulting TMD affinity as the partonic kinematics differ from those utilized in the original study. Because the original study had implemented a basic guess for their partonic variables rather than simulating them, their results did not match. Given that MC simulation provides as close of a look at the partonic kinematics as is accessible, we are working to implement the partonic information I extracted from the MC into the affinity tool to provide a more realistic model.  



## Personal Experience essay

`Please describe your experience with and overcoming a significant obstacle in pursuing your educational or research goals.`

Ideas:

1. CS 290 last semester
2. Math 122
3. Modern Physics
4. Research
   1. Affinity project
      1. Fighting through slump / not knowing how to progress, continuing
5. **"Pivoting affinity project from original Neural network method to BOX method and from dihadrons to single hadrons etc"**



### Outline

1. Thesis
   1. I faced many obstacles when trying to produce results during my first research project, but found that getting help from more experienced researchers and keeping working hard allowed me to persevere and see the bigger picture
      1. Many of my attempts did not work really but I had to overlook the "lost" time I spent on that and instead focus on the bigger picture
2. Anecdotes/obstacles
   1. Beginning: no knowledge on any of the physics involved, and the project was kind of complex
      1. Getting started took a while and I was not sure how to actually do the research as I was not yet familiar with any part of research
   2. Middle:
      1. Not sure how to interpret the results I was getting
      2. Many small things that I didn't know how to do/what they were but I could have just asked as all the grad students/advisor knew
      3. Sometimes spent many hours or even weeks on certain aspects of the project but had to abandon them as those parts were not entirely relevant
   3. Resolution / overcoming
      1. Realized I need to check in more often with advisor and other collaborators to ensure we are on the same page and that I am not tunnel visioning
      2. Finally am reaching a place where we have found where the differences in our techniques arise from
      3. Have narrowed investigation
         1. Although this means I have to abandon some of the work I did, like the dihadron studies, we have results that are interesting and useful that we can focus on
            1. May be able to return to the dihadrons later but for now cannot just try to make something from it just for the sake of salvaging that work



### First draft



#### Intro

​	My experience with physics research began the summer of my freshman year when I started working on a project focused on estimating the applicability of different factorization theorems in different phase space for deep inelastic scattering experiments. Although the project was my introduction to research, I worked on it mostly independently with help from my advisor and member's of my research group when needed. Throughout the project I found myself quite confused and unsure of how to proceed many times, but I persevered and kept working towards results, helping shape me as a research in addition to leading to some interesting and useful findings.

​	Starting the project after my first year of undergraduate physics was daunting as I had no experience with the subject of deep inelastic scattering or nuclear physics as a whole, and I had no idea what research actually looked like. Nevertheless, I was quite excited to learn more both about nuclear physics and the world of physics research, and I worked hard to get up to speed enough to understand what the goal of my project was meant to be. In addition to the technical skills I needed to foster, mainly programming, I had never really spent 40 hours a week working on anything that wasn't school, meaning I had to learn how to work independently for many hours a day while figuring out what steps I needed to take to further my investigation. My advisor and the graduate students around me were very helpful but I did not ask for help as much as I should have, opting instead to try to figure things out myself. Unfortunately, I had more things to figure out than I could have known, so it took me longer than I had expected to begin actually get to a place where I could make calculations and produce results. I had learned through my courses to read my course textbooks and learn that way, but I was finding that the research papers relevant to my study were not nearly as digestible as my introductory textbooks. While all of these obstacles may sound damaging to my research goals, I believe that many of them have helped me grow as a student and researcher. A prime example of this grow is how I learned some basic programming skills as a necessity to begin my project, which has lead me to take several computer science courses and turn programming into something I really enjoy. 

​	<topic sentence>. Looking back through my code base for my first project, I am amazed to see how much code I wrote and how much work I did that did not directly progress the final results. Many times throughout the project I found that the previous hour's, day's, or week's work was no longer useful after coming to a realization or meeting with collaborators. Sometimes I made mistakes, but often I did exactly what I planned to do only for our goals to pivot afterwards. The first major pivot came when I met with a Professor who had worked on the original proposal of the model I was using. Before the meeting, I had been working solely based on a research paper by the Professor's group, and had to infer many things due to my limited knowledge of the subject. During our meeting, I realized that my entire method of calculating the quantity of interest differed from the original researchers' intent. At first I had felt that I had messed up, completely missing the goal by misreading their paper. However, we found that my method had some qualities that would make for a more interesting study than otherwise would have been, at least with a few tweaks to get back on track. So, yes I had created a program that did not correctly produce the results that we wanted, but by meeting with that Professor I was able to refocus my research goals and adjust my work to be much better than before. I had to abandon some code that no longer made sense to use, but I found that this was a necessary part of the process, and that progress on more independent projects often required some risks and mistakes.

​	Today, currently working on my third nuclear physics research project, I am much more comfortable asking for help and making mistakes. I believe that if my initial project had gone much smoother I would not have learned as much as I did, and hence I am a better researcher because of it. Furthermore, the hard work has made the results I have produced even more rewarding as it has shown me that I can succeed if I keep working despite how daunting research can seem at times.
