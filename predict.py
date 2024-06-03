import transformers
from transformers import pipeline
from data_loader import title_dataset

"""TEST ON EXAMPLE"""

text = title_dataset['test'][5]['text']
text = """
ve existed in my plus-size body for 27 years now, and I've pretty much accepted it—except for my double chin. Maybe it's because I'm a beauty editor who's constantly looking at her face as I'm testing the latest glowy foundations and face serums (and taking a bajillion selfies to prove it). But the extra fat and puffy look of my chin is my cross to bear. And I've tried every single thing out there to fix it (trust me). The single only thing that's ever helped me get rid of my double chin was chin liposuction, specifically AirSculpt.

Before I detail my AirSculpt experience, you've gotta grasp why I decided just to go the route of liposuction, because it's definitely the most expensive, invasive, and—let's be real—intense option. I'll be the first person to shout about self-love and body confidence from the rooftops too. There is nothing wrong with having a lil fat under your chin—never, ever, ever—but I didn't love mine and constantly looked for treatments, which included:

CoolSculpting: This works by freezing your fat cells until they die, but involved months of painful, swollen downtime.
KyBella: A series of injections of a compound that kills fat cells, which was just as painful as it sounds and only gave me subtle results.
Chin filler: My doctor injected a tiny bit of hyaluronic-acid filler into my tip of my chin, which added a little projection to slightly changed the shape of my chin, but couldn't disguise the fat.
Gua sha and jade rolling: Facial massages with tools like gua sha and jade rollers can help encourage lymphatic drainage to temporarily slim your face, but it didn't lead to lasting results nor magically get rid of fat.
Jaw filler: Similar to chin filler, a doctor injected filler into my jawline to add some definition and create a sharper appearance, but this didn't get rid of or hide my double chin.
NuFace and microcurrent facials: Microcurrent works by emitting an electrical current into your muscles to "tone" them, giving my round face a more contoured, look, but didn't get rid of fat (seeing a pattern here?).
Facial yoga: These facial movements are said to help exercise and tone your face, but because you can't target fat loss, this did absolutely nothing (lol, sry).
Depuffing creams: These creams are filled with anti-inflammatories, like arnica, allantoin, caffeine, and aloe vera to help depuff your skin and calm down inflammation, but didn't disguise fat.
Electrical muscle stimulation (EMS), like EmFace: Using electricity to mimic natural facial contractions and strengthen my muscles, this gave my face a noticeable lift for a few hours but didn't improve my double chin.
Losing weight: Of course, this slimmed my face a bit, but it barely touched my double chin, which is genetic.
When all of the above couldn't give me a sharp, Bella Hadid-esque jawline, I knew the only way to get the results I yearned for was to go the most invasive route. And I'm here to announce: Yup, laser liposuction, specifically AirSculpt, got rid of my double chin. Finally.

"""


text_2 = """
think we can all agree Hailey Bieber is pretty much winning this summer. As you may or may not have heard, the Rhode founder and her husband Justin are currently expecting Baby Bieber. (Yes, “That Should Be Me” is still playing on a loop.) Translation? Hailey has been looking as glowy and radiant as ever. But it turns out she also has a hack to the beachy bronze glam she’s been sporting lately. In a recent GRWM video on her YouTube channel, Hailey revealed she’s been using a $28 Dolce Glow Contour Self-Tanning Sculptor, and guys, I’ve never ran to buy a product so fast.

Whenever Mrs. Bieber drops a makeup tutorial, I immediately sit up straight and take notes. After all, she is the queen of glazed donut skin. In her video, Hailey listed out every product she’s been using as her “go-to” makeup lately. And it’s surprisingly full of affordable products! After prepping her skin with her signature Rhode products (the Glazing Milk and Peptide Glazing Fluid), Hailey went straight in with the self-tanning wand for an underpainting approach.

Contour Self-Tanning Sculpt + Glow. Once she applies the contour along her cheekbones and the top of her forehead, Hailey blends it all in with the Hourglass Ambient Soft Glow Foundation Brush. While using self-tanner on your face to achieve a contoured look is nothing new, I love that Hailey has added a rosy twist to her glam, which she’s calling a “peachy beachy” makeup look. You’ll often find that the faux glow technique usually involves products that are jam-packed with skincare-forward ingredients, which helps with the whole super glowy summer skin look and feel. And in the case of Dolce Glow’s sculptor, it’s infused with four types of hyaluronic acid and DHA which help create a gradual tan. More than anything, it is super hydrating and allows you to look freshly bronzed without lying in the sun for hours.


"""


text_2 = "summarize: " + text_2
print(text_2)

summarizer = pipeline("summarization", model="t5_small_title_generator")

summary_output = summarizer(text_2, max_length=50, min_length=10)

generated_title = summary_output[0]['summary_text']

if '.' in generated_title:
    generated_title = generated_title[:generated_title.index('.') + 1]

print(generated_title)

