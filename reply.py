import numpy as np
from rake_nltk import Rake

def fltr(ranks,kys):
    count=[]
    
    for v in kys.values():
        n=0
        for l in ranks:
            if l in v:
                n+=1
        count.append(n)
        
    idx=np.argmax(np.array(count))    
    return idx

def listener(inp):
    r = Rake()
    texts=["It has been 12 days since I received the product, but the product gives an error. It doesn't work properly. What can I do",
           "My order was delivered to the wrong address. What can I do?",
           "After adding the product to the cart, its price increased, why?",
           "How many installments do you pay for furniture, to which bank?",
           "The equipment/accessories of the product I sent to the service were missing?",
           "Will I Pay for Shipping?",
           "My order is late, what should I do?"]

    a=["Our products are shipped after all checks have been made in our store. You can contact our authorized service. If the product is not a user error, we will provide assistance as necessary.",
    "First of all, the cargo must be returned. You can order again by updating the address from the virtual store. ",
    "Prices of the products are updated daily.Adding to cart doesnt fix the price.",
    """Yapı Kredi 9
    akbank 12
    job bank 10
    public bank 15""",
    "If you send us a photo of the box of the product, we can send the missing item. ",
    "For purchases under 750 TL, you will be charged a shipping fee. We wish you a nice day.. ",
    "Good day, sir, if you share the order number of the product with us, we can provide information about the relevant cargo company and the status of your order. Have a nice day.. "]

    kys=dict()
    kvp=dict()
    for id,x in enumerate(texts):
        r.extract_keywords_from_text(x)
        ll=r.get_ranked_phrases_with_scores()[:6]
        ple=tuple([lx[1] for lx in ll])
        kys.update({id:ple})
        kvp.update({ple:a[id]})


    r.extract_keywords_from_text(inp)
    tr=r.get_ranked_phrases_with_scores()[:10]

    pul=tuple([lx[1] for lx in tr])
    try:
        ca=kvp[pul]
    except:
        id=fltr(pul,kys)
        ca=kvp[kys[id]]

    return ca    