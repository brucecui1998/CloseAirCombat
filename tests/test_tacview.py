import socket
import threading
import time

class tacview(object):
    def __init__(self):
        host = input("请输入服务器IP地址：")
        port = int(input("请输入服务器端口："))

        # 提示用户打开tacview软件高级版，点击“记录”-“实时遥测”
        print("请打开tacview软件高级版，点击“记录”-“实时遥测”，并使用以下设置：")
        print(f"IP地址：{host}")
        print(f"端口：{port}")

        # 创建套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_socket.bind((host, port))

        # 启动监听
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        # 等待客户端连接
        client_socket, address = server_socket.accept()
        print(f"Accepted connection from {address}")

        self.client_socket = client_socket
        self.address = address

        # 构建握手数据
        handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
        # 发送握手数据
        client_socket.send(handshake_data.encode())


        # 接收客户端发送的数据
        data = client_socket.recv(1024)
        print(f"Received data from {address}: {data.decode()}")
        print("已建立连接")

        # 向客户端发送头部格式数据

        data_to_send = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                        "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
                        )
        client_socket.send(data_to_send.encode())

    def send_data_to_client(self, data):

        self.client_socket.send(data.encode())

if __name__ == "__main__":
    tacview = tacview()
    data_list = [
        "#0.00\nA0100,T=119.99999999999999|59.999999999999986|8902.421354242131|5.124908336161374e-15|2.6380086088911072e-15|92.1278924460462,Name=F16,Color=Red\n"
        "#0.20\nA0100,T=120.00056284766772|59.99998954916791|8902.214730471585|1.3946936905659462|-0.37548687780585166|92.2068688527138,Name=F16,Color=Red\n",
        "#0.40\nA0100,T=120.00112753643519|59.99997905381184|8901.50075940655|3.3309163259433774|-1.5022126375785063|92.20565433550017,Name=F16,Color=Red\n",
        "#0.60\nA0100,T=120.00169233145378|59.99996851211707|8900.424199637086|2.786130884419372|-1.9402045078395926|92.14016570096325,Name=F16,Color=Red\n",
        "#0.80\nA0100,T=120.00225851944843|59.999957948547845|8898.993838468623|2.309544270950709|-3.101580460265669|92.1595817499671,Name=F16,Color=Red\n",
        "#1.00\nA0100,T=120.00282829116189|59.99994735308507|8897.110349190063|3.430543655959103|-5.470428357694592|92.23048140982122,Name=F16,Color=Red\n",
        "#1.20\nA0100,T=120.00340214201114|59.999936882785185|8894.481743723774|8.221415272395108|-8.305702411429808|92.28932565770451,Name=F16,Color=Red\n",
        "#1.40\nA0100,T=120.0039800887564|59.999926938002936|8890.91136290946|16.445961110534416|-11.209905844226334|92.0945084923848,Name=F16,Color=Red\n",
        "#1.60\nA0100,T=120.00456172870842|59.999918234679306|8886.343155609236|27.929869533166514|-13.974191189082877|91.42821828812505,Name=F16,Color=Red\n",
        "#1.80\nA0100,T=120.00514669603349|59.999911781651086|8880.771636800659|42.469922621047196|-16.16037233106113|90.30241357163952,Name=F16,Color=Red\n",
        "#2.00\nA0100,T=120.0057330855506|59.9999083361402|8874.329358054087|59.26658028141615|-16.43138047324893|90.16032232966084,Name=F16,Color=Red\n",
        "#2.20\nA0100,T=120.00631927147913|59.99990716108694|8867.268477978767|73.64306413139174|-14.850358705766423|92.55245229594057,Name=F16,Color=Red\n",
        "#2.40\nA0100,T=120.0069055290387|59.999904690042584|8859.8229020421|80.29289312005385|-12.893267974256315|96.9178461713516,Name=F16,Color=Red\n",
        "#2.60\nA0100,T=120.00749095159223|59.99989660512858|8852.103072312784|80.36426479476215|-11.456431123763121|101.55132375341236,Name=F16,Color=Red\n",
        "#2.80\nA0100,T=120.00807399130964|59.99987950890545|8844.186396074278|75.28196618620005|-10.45126951891485|105.48514896022623,Name=F16,Color=Red\n",
        "#3.00\nA0100,T=120.00865409629424|59.99985271710337|8836.162429316622|68.71775360226547|-9.615275956996049|108.63267356674214,Name=F16,Color=Red\n",
        "#3.20\nA0100,T=120.00923195391493|59.999817829715205|8828.061268871892|64.57258267308141|-8.773463530383811|111.20804434329088,Name=F16,Color=Red\n",
        "#3.40\nA0100,T=120.0098079052632|59.99977570659703|8819.901945315627|62.22027080888067|-7.873479831149966|113.32321968406606,Name=F16,Color=Red\n",
        "#3.60\nA0100,T=120.01038112818262|59.99972559963645|8811.772571508523|59.45154332233721|-6.961251542949214|115.0198537855895,Name=F16,Color=Red\n",
        "#3.80\nA0100,T=120.01095023098337|59.99966607973344|8803.833312539158|54.1277621804789|-6.0202551087621075|116.35368931856245,Name=F16,Color=Red\n",
        "#4.00\nA0100,T=120.01151489797557|59.99959725513199|8796.201178178082|47.25413821719767|-4.923189635695278|117.46393717976622,Name=F16,Color=Red\n",
        "#4.20\nA0100,T=120.0120754634765|59.99952007927246|8788.96219394974|39.96869936147549|-3.555436603424211|118.38263183215024,Name=F16,Color=Red\n",
        "#4.40\nA0100,T=120.01263234065208|59.99943560638457|8782.196505322105|33.153902253148914|-1.893261889584092|119.0503243624761,Name=F16,Color=Red\n",
        "#4.60\nA0100,T=120.01318604913719|59.99934498976197|8775.95295088049|29.07410498782518|-0.022530652457955918|119.50125733693001,Name=F16,Color=Red\n",
        "#4.80\nA0100,T=120.01373723470591|59.99924943649844|8770.168035616789|29.05717012307691|1.9227870535419862|119.87023826832788,Name=F16,Color=Red\n",
        "#5.00\nA0100,T=120.01428556158824|59.99914913160295|8764.870568604425|29.518808190167356|3.7639218287736242|120.19198048338609,Name=F16,Color=Red\n",
        "#5.20\nA0100,T=120.01483008182397|59.999043836683754|8760.179587978515|27.551022857864023|5.384113112222622|120.52428693059758,Name=F16,Color=Red\n",
        "#5.40\nA0100,T=120.01537054973726|59.998933945767206|8756.128760551035|23.36918060810404|6.866509936588442|120.88003969190424,Name=F16,Color=Red\n",
        "#5.60\nA0100,T=120.01590712936854|59.998820166747194|8752.7306789184|17.970069668951847|8.34491301629079|121.13304341795302,Name=F16,Color=Red\n",
        "#5.80\nA0100,T=120.01644018093454|59.998703344746765|8749.997455435172|12.379821527515398|9.884387901682123|121.17917807516383,Name=F16,Color=Red\n",
        "#6.00\nA0100,T=120.01697016377796|59.99858435355848|8747.929532850034|7.335001983977035|11.280694558004738|120.92966504521263,Name=F16,Color=Red\n",
        "#6.20\nA0100,T=120.01749766007771|59.99846400727184|8746.487337636012|4.4491871367529185|12.322982864353904|120.43668007002744,Name=F16,Color=Red\n",
        "#6.40\nA0100,T=120.01802318656249|59.998342826884745|8745.561738879393|4.530933019633621|12.93081923430023|119.84674020984188,Name=F16,Color=Red\n",
        "#6.60\nA0100,T=120.01854668670903|59.99822103188242|8745.07416324098|5.568631733634867|11.702179218627192|119.13297925910396,Name=F16,Color=Red\n",
        "#6.80\nA0100,T=120.01906861769154|59.998098631136926|8744.906837770066|6.209784592996896|9.142852086055004|118.37720146286705,Name=F16,Color=Red\n",
        "#7.00\nA0100,T=120.01958970565781|59.997975633679125|8744.900884926663|5.9373884627326206|6.284923095091635|117.69191058575662,Name=F16,Color=Red\n",
        "#7.20\nA0100,T=120.02011143113569|59.99785182346219|8744.96773777509|3.7492715056405372|4.665147313596463|117.20670583015792,Name=F16,Color=Red\n",
        "#7.40\nA0100,T=120.02063532576791|59.997727020045474|8745.09577357707|-1.6713018552477703|2.7113551063215118|116.79551375025454,Name=F16,Color=Red\n",
        "#7.60\nA0100,T=120.02116084275673|59.99760177065212|8745.129062050948|-6.5866815370394365|-0.13587133908262744|116.72485710887213,Name=F16,Color=Red\n",
        "#7.80\nA0100,T=120.02168775306855|59.99747615034484|8744.819262704712|-6.644388199258331|-2.9792538454254673|117.02491165957372,Name=F16,Color=Red\n",
        "#8.00\nA0100,T=120.02221616585177|59.997349890741596|8743.983044855708|-2.529427235025957|-3.913146377766713|117.22675683833549,Name=F16,Color=Red\n",
        "#8.20\nA0100,T=120.02274523375476|59.99722320566088|8742.62421165002|4.252440208195821|-2.06846232581264|117.27695990356268,Name=F16,Color=Red\n",
        "#8.40\nA0100,T=120.02327502340762|59.99709603494164|8740.987105403236|11.588414824106492|1.6805884601378314|117.54606412245094,Name=F16,Color=Red\n",
        "#8.60\nA0100,T=120.02380519618228|59.9969680245525|8739.29617899272|19.222712001840634|6.140130614184847|118.37845588025964,Name=F16,Color=Red\n",
        "#8.80\nA0100,T=120.02433468508555|59.996838552849496|8737.736945713405|24.26717361044226|9.919342466229978|119.63322000190706,Name=F16,Color=Red\n",
        "#9.00\nA0100,T=120.02486149901097|59.99670682543237|8736.547129092023|22.76816078188761|12.741132751174806|120.92795824206215,Name=F16,Color=Red\n",
        "#9.20\nA0100,T=120.02538417905569|59.996572511041045|8735.940405517558|16.275152973467865|14.800522263708057|121.96277360748982,Name=F16,Color=Red\n",
        "#9.40\nA0100,T=120.02590261258938|59.99643620600137|8736.006566492111|7.686208434139324|16.15335607886926|122.49608737857409,Name=F16,Color=Red\n",
        "#9.60\nA0100,T=120.02641753341585|59.99629899145001|8736.724950718914|-0.38325923979979387|16.389524498016726|122.44550613264609,Name=F16,Color=Red\n",
        "#9.80\nA0100,T=120.02692967834268|59.99616192135361|8737.991955948419|-7.083424546872841|14.657936817014603|122.03177121967542,Name=F16,Color=Red\n",
        "#10.00\nA0100,T=120.0274405202835|59.99602548443802|8739.618806107925|-10.166855909249001|11.745230235161353|121.70666448542134,Name=F16,Color=Red\n",
        "#10.20\nA0100,T=120.02795127371243|59.99588936428511|8741.316490013856|-6.810625509798238|8.397012625122443|121.59330866177768,Name=F16,Color=Red\n",
        "#10.40\nA0100,T=120.02846261782156|59.99575289163477|8742.80840964211|1.1225425533379993|4.955289714303088|121.46741777596188,Name=F16,Color=Red\n",
        "#10.60\nA0100,T=120.02897502993663|59.99561571267831|8743.88482625852|9.22706079462471|1.5544257558371462|120.97618055496477,Name=F16,Color=Red\n",
        "#10.80\nA0100,T=120.02948839832199|59.99547792455058|8744.524879545943|12.436215325168337|-1.7363458162990288|120.03789843366225,Name=F16,Color=Red\n",
        "#11.00\nA0100,T=120.03000216315044|59.99533978023669|8744.743018293133|11.214418439024373|-4.810308654748249|118.95625941886395,Name=F16,Color=Red\n",
        "#11.20\nA0100,T=120.03051575555381|59.995201564595746|8744.538376355651|5.8995926922416375|-7.3216855884750665|118.05478492915104,Name=F16,Color=Red\n",
        "#11.40\nA0100,T=120.03102866675346|59.995063670057405|8743.85951111473|-0.07956038768394066|-7.507195201020067|117.80890654303441,Name=F16,Color=Red\n",
        "#11.60\nA0100,T=120.03154134298092|59.994925980464785|8742.69617986799|-1.584799554902888|-4.918600185704944|118.01751334109915,Name=F16,Color=Red\n",
        "#11.80\nA0100,T=120.03205487823631|59.994788079640244|8741.111752349569|0.9986778406479817|-1.1157006086244188|118.31358499068168,Name=F16,Color=Red\n",
        "#12.00\nA0100,T=120.03256961456611|59.99464978747803|8739.279795991311|5.950884365266184|3.380187088707772|118.73474123547759,Name=F16,Color=Red\n",
        "#12.20\nA0100,T=120.03308509325463|59.99451088410973|8737.459113473664|11.919835448981488|7.867621755801504|119.43892138138821,Name=F16,Color=Red\n",
        "#12.40\nA0100,T=120.03360015545749|59.99437096714468|8735.862985423744|15.478762605061473|11.460820340845764|120.32819271994498,Name=F16,Color=Red\n",
        "#12.60\nA0100,T=120.03411328006972|59.9942296016528|8734.730232172931|12.9471473500549|14.108117553218383|121.17028772192057,Name=F16,Color=Red\n",
        "#12.80\nA0100,T=120.03462342306138|59.99408691043464|8734.255146015636|6.059596848292941|15.775269289040285|121.73035512538273,Name=F16,Color=Red\n",
        "#13.00\nA0100,T=120.03513072973315|59.99394370974663|8734.46110964184|-2.3320115939086206|16.0921198610651|121.87188517481084,Name=F16,Color=Red\n",
        "#13.20\nA0100,T=120.03563632337578|59.99380086343215|8735.237877890111|-10.08838576325455|14.37015456339123|121.77066209106229,Name=F16,Color=Red\n",
        "#13.40\nA0100,T=120.03614160030135|59.99365887009206|8736.400618078951|-14.247460682317962|11.47275145323707|121.80484821272177,Name=F16,Color=Red\n",
        "#13.60\nA0100,T=120.03664723466022|59.99351745452802|8737.649744317103|-11.772174317739672|8.143918008141668|122.05121352869259,Name=F16,Color=Red\n",
        "#13.80\nA0100,T=120.03715361420954|59.993375817060425|8738.70556024805|-4.504199805381417|4.714336273099684|122.27581225250307,Name=F16,Color=Red\n",
        "#14.00\nA0100,T=120.0376609912493|59.99323348680339|8739.358148655609|5.311438282506434|1.286748495215471|122.20378927860054,Name=F16,Color=Red\n",
        "#14.20\nA0100,T=120.03816908132589|59.993090516282344|8739.594937351138|13.326256141391132|-1.7273338559816214|121.62411425633074,Name=F16,Color=Red\n",
        "#14.40\nA0100,T=120.03867706020307|59.992946969710985|8739.508991814771|16.336282174757628|-2.5123008384021506|121.09847388359978,Name=F16,Color=Red\n",
        "#14.60\nA0100,T=120.03918395465864|59.99280272149844|8739.273481436474|14.856089169865049|-0.3089771057940178|121.2867555215008,Name=F16,Color=Red\n",
        "#14.80\nA0100,T=120.03968972970632|59.99265751281636|8739.05696766277|9.80251912392855|3.8841331599308635|121.93671356280376,Name=F16,Color=Red\n",
        "#15.00\nA0100,T=120.04019454373388|59.99251138014739|8739.03099647027|2.6704423577805483|8.386170297509082|122.39849323372054,Name=F16,Color=Red\n",
        "#15.20\nA0100,T=120.04069839011193|59.99236487188692|8739.33493921541|-4.952513460280964|11.707827822289051|122.36124316154672,Name=F16,Color=Red\n",
        "#15.40\nA0100,T=120.04120124903973|59.992218786660025|8739.972919597247|-10.087189176214467|12.131120885383538|122.1812690815912,Name=F16,Color=Red\n",
        "#15.60\nA0100,T=120.04170394341057|59.992073179574724|8740.763719115701|-9.624242570594332|10.218133530800964|122.23391454593266,Name=F16,Color=Red\n",
        "#15.80\nA0100,T=120.04220719526508|59.9919275006036|8741.468785399738|-3.692259598759761|7.336783937362165|122.3782467923045,Name=F16,Color=Red\n",
        "#16.00\nA0100,T=120.04271129750616|59.99178127695004|8741.912025690186|3.5639558164114233|4.111983085824899|122.2933614697391,Name=F16,Color=Red\n",
        "#16.20\nA0100,T=120.04321602525259|59.99163446903769|8742.078014772605|7.146579352361522|0.789578151409119|121.83602101305695,Name=F16,Color=Red\n",
        "#16.40\nA0100,T=120.04372107256484|59.99148717703599|8741.969631823773|6.347830099309232|-2.5377537006596493|121.16863412505427,Name=F16,Color=Red\n",
        "#16.60\nA0100,T=120.04422615740911|59.99133965900534|8741.548212380729|2.9605827844585426|-5.4476955064976265|120.64291438023493,Name=F16,Color=Red\n",
        "#16.80\nA0100,T=120.04473131781272|59.9911921358161|8740.669743247596|1.2121773217987872|-6.2028418041280275|120.52139835068498,Name=F16,Color=Red\n",
        "#17.00\nA0100,T=120.04523676937178|59.991044561128994|8739.275506306065|3.506233610400748|-3.947559149676192|120.7081961062522,Name=F16,Color=Red\n",
        "#17.20\nA0100,T=120.04574307085471|59.99089668204073|8737.490935600135|8.229219061720862|0.48353481029707074|121.21103868549865,Name=F16,Color=Red\n",
        "#17.40\nA0100,T=120.0462500014689|59.99074810770642|8735.610548621313|12.392288465721656|5.512795354579887|122.00820643897563,Name=F16,Color=Red\n",
        "#17.60\nA0100,T=120.0467562612053|59.990598377023595|8733.974305800659|12.576828941341239|9.754328398219855|122.84224585106814,Name=F16,Color=Red\n",
        "#17.80\nA0100,T=120.04726045529614|59.990447311491|8732.86866964919|8.357286779853618|12.936376293925566|123.47881541629926,Name=F16,Color=Red\n",
        "#18.00\nA0100,T=120.04776223345986|59.99029531985381|8732.44127242899|1.40178911914434|15.290815130379222|123.76839920145895,Name=F16,Color=Red\n",
        "#18.20\nA0100,T=120.048261789807|59.99014334222705|8732.716447713357|-6.097601269266987|16.736558817862395|123.65408399077447,Name=F16,Color=Red\n",
        "#18.40\nA0100,T=120.04875935564455|59.98999240542524|8733.60578212056|-12.530930425483632|15.938453194774292|123.38562951058766,Name=F16,Color=Red\n",
        "#18.60\nA0100,T=120.0492561405814|59.989842880084424|8734.935966842133|-14.907589692639057|13.467352955859033|123.33410668374836,Name=F16,Color=Red\n",
        "#18.80\nA0100,T=120.04975309745396|59.989694210856776|8736.428629947899|-10.511413340332442|10.283832072078113|123.50778952517511,Name=F16,Color=Red\n",
        "#19.00\nA0100,T=120.05025061134918|59.989545477047585|8737.81173788918|-1.4975146721396004|6.874414628606804|123.63821113852285,Name=F16,Color=Red\n",
        "#19.20\nA0100,T=120.0507490196385|59.98939608463186|8738.871457273499|7.597903850616895|3.4484398620075445|123.37935110466779,Name=F16,Color=Red\n",
        "#19.40\nA0100,T=120.05124813507618|59.98924602244505|8739.57777812758|11.744911777758645|0.1064931541480969|122.57739646745719,Name=F16,Color=Red\n",
        "#19.60\nA0100,T=120.05174805654218|59.98909533016561|8739.944295902682|11.39318998722254|-2.8470109749852752|121.5879847680932,Name=F16,Color=Red\n",
        "#19.80\nA0100,T=120.05224817594632|59.98894424016393|8740.022226014873|6.815972950572367|-3.74811537969169|121.04861377908014,Name=F16,Color=Red\n",
        "#20.00\nA0100,T=120.05274778434675|59.98879312743987|8739.845601478342|0.9409831352151766|-1.7053530794429757|121.01520387907783,Name=F16,Color=Red\n",
        "#20.20\nA0100,T=120.0532477748078|59.98864187788115|8739.445919548398|-1.559749723560069|2.293895079528707|121.11848942023535,Name=F16,Color=Red\n",
        "#20.40\nA0100,T=120.05374880310384|59.988490338251246|8738.895626670546|0.5095683725121597|6.472917296833313|121.26302595100496,Name=F16,Color=Red\n",
        "#20.60\nA0100,T=120.05425059765085|59.988338447461196|8738.363501563204|2.6809153639023906|10.153609768725667|121.40116566268702,Name=F16,Color=Red\n",
    ]
    for data in data_list:
        tacview.send_data_to_client(data)
        time.sleep(0.5)  # 根据需要调整延迟以模拟实时数据流