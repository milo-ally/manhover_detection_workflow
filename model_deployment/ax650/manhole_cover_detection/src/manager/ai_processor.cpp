#include "ai_processor.h"
#include "../../utilities/sample_log.h"
#include "../include/osd_renderer_interface.h"
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include "ax_sys_api.h"

// 闂備礁鎼鍛偓姘嵆閸┾偓妞ゆ帒鍊稿瓭闂佹悶鍊濇禍璺侯嚕椤愩倖鏆滄い鏂垮⒔閻涖儵姊洪崫鍕ユい顐㈩槸閻ｅ灚鎷呴懖婵堝枛閸ㄦ儳鐣烽崶锝呬壕濡炲瀛╁畷澶嬨亜閺嶃劍鐨戠紒鎰仱閺屾盯寮崒姣款剚绻涢弶鎴烆潒I婵犵妲呴崹顏堝焵椤掆偓绾绢參鍩€椤掑倸鍘撮柡浣哥Т閳荤晫绮婇崲绶€lName 闂備焦妞垮鍧楀礉瀹ュ钃熼柕鍫濐槸绾偓闂侀€炲苯澧寸€?crowd vs human闂備焦瀵х粙鎴︽偋閸℃稑鐭楅柨鐔哄У閸嬨劑鏌曟繛鍨姕闁稿﹤宕埥澶愬箻椤栨矮澹?path 闂備礁鎼幊搴ㄥ磻婵犲嫭顫?AIProcessor::AIProcessor(const std::string& modelPath, const std::string& modelName,
                         const nlohmann::json& modelParams) {
    if (!modelPath.empty()) {
        loadModel(modelPath, modelName, modelParams);
    }
}

// 闂備礁鎼鍡涘礉閺嶎厽鍊垫い鏍仜缁€鍕煠閹帒鍔滄繛鍫濈埣閺屻劌鈽夊▎鎺戭棟閻庤鎸哥换鎴濐焽椤忓棙顫曠紒鎰哺娣囧﹪顢涘璇蹭壕鐎规洖娲犻崑鎾绘倻閼恒儲娅栭悗鍏夊亾闁告洦鍋呴悵鐢告⒑鐞涒€充壕闂佺顫夐崝妤呭Χ椤愶絿绡€?
AIProcessor::~AIProcessor() {
    unloadModel();
}

// 闂備礁鎲″缁樻叏閹灐褰掑床閸㈡柨鈹戦埥鍡楃仚闁逞屽墮绾绢參鍩€椤掑倸鍘存鐐存崌楠炴帒顓奸崱鈺佷缓闂佸湱鍘ч悺銊╁箰閹间礁绠柡澶庮嚦閻旂厧鐏崇€规洖娲犻崑鎾搭槹鎼存ê浜鹃柛锔诲幖娴犳帞绱掗煬鎻掑姦闁轰礁绉撮埢鐣岀矈閸㈢穩lName 闂備焦妞垮鍧楀礉瀹ュ钃熼柕鍫濐槸缁犵敻鏌熼崫鍕ｆい鎰矙濮婃椽顢楅埀顒勫箟閿熺姴绠瑰ù鐓庣摠閺咁剟鏌涜濠狗wd 闂?human 闂備胶顭堢换妤佺椤掑嫬鏋?path 闂備礁鎼幊搴ㄥ磻婵犲嫭顫?bool AIProcessor::loadModel(const std::string& modelPath, const std::string& modelName,
                            const nlohmann::json& modelParams) {
    std::lock_guard<std::mutex> lock(modelMutex_);
    
    // 闂備胶顭堢换鎰版偋閸℃鏆ら煫鍥ㄦ⒐婵粓鏌﹀Ο渚Ц闁搞倖甯￠弻娑㈠籍閸屾顒佺箾閺夋垶鍠橀柟顖氬暣瀹曠喖顢栭悢鍓佺獢鐎?
    if (model_) {
        unloadModel();
    }

    if (modelPath.empty()) {
        return false;
    }

    modelName_ = modelName;
    modelParams_ = modelParams.is_object() ? modelParams : nlohmann::json::object();
    osdRenderer_.reset();
    // 闂備礁婀辩划顖滄暜婵犲倵鏋?OSD 闂傚倷绶￠崜娆撳箟閿熺姴绠瑰ù鐓庣摠閺咁剚鎱ㄥ鍡楀箺缂?modelName 闂備礁鎼幊搴ㄥ磹婵犳艾鏋?modelName闂備焦瀵х粙鎴︽偋閸涱厜褎寰勯幇顑跨炊闂傚嫬娲︾粋鎺戔槈閵忊檧鎸呴梺绯曞墲閼归箖鎮楅悡骞熺懓顭ㄩ崘鈺傚創闂佺濮ら〃濠傜暦閹达箑鍨傛い鎰╁劚婵矂姊虹粙璺ㄧ濞存粠鍓熷?path闂?
    const std::string& pluginHint = modelName_.empty() ? modelPath : modelName_;
    applyModelParamsToEnv(pluginHint, modelParams_);

    // 闂備礁鎼粔鐑斤綖婢跺﹦鏆?pluginHint 闂備胶鍘ч〃搴㈢濠婂嫭鍙忛柍鍝勬噺閻掕顭跨捄渚剰妞ゅ繈鍎甸弻鐔虹磼濡櫣鐟愬┑鐐叉閸ㄦ椽骞?
    std::string pluginPath = "./libmanhole_plugin.so";
    ALOGN("[AIProcessor] Using manhole-cover plugin for model: %s", modelPath.c_str());

    std::ifstream test_file(modelPath);
    if (!test_file.good()) {
        ALOGE("Model file does not exist: %s", modelPath.c_str());
        return false;
    }
    test_file.close();

    // 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓缁犵敻鏌熼崫鍕ｆい鎰矙楠炴牜鈧稒蓱閹牏绱掓潏銊ф噰鐎规洘绻堟俊鎼佹晜閻熼澹曟繛杈剧到閹芥粌顫濋妸鈺傜叆?
    dlerror();
    void* handle = dlopen(pluginPath.c_str(), RTLD_LAZY);
    if (!handle) {
        ALOGW("[AIProcessor] dlopen %s failed, trying ./bin/libmanhole_plugin.so", pluginPath.c_str());
        handle = dlopen("./bin/libmanhole_plugin.so", RTLD_LAZY);
        if (!handle) {
            ALOGE("[AIProcessor] dlopen manhole plugin failed: %s", dlerror());
            return false;
        }
    }

    // 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉墮缁€鍡樼箾閹寸儐鐒界紒鎲嬮檮娣囧﹪顢涘璇蹭壕鐎规洖娲犻崑鎾搭槹鎼存ê浜鹃柛锔诲幖娴犳帞绱掗煬鎻掑姦闁诡垰鍟村畷鐔碱敃閵忥紕妲梻浣芥〃閼宠埖鏅跺Δ鍐ｅ亾閸偄鍝洪柡?
    CreateAIModelFunc create = (CreateAIModelFunc)dlsym(handle, "CreateAIModel");
    if (!create) {
        ALOGE("dlsym CreateAIModel failed: %s", dlerror());
        dlclose(handle);
        return false;
    }

    // 闂備礁鎲＄敮妤冪矙閹寸姷纾介柟鐐窞閻旂厧鐏崇€规洖娲犻崑鎾搭槹鎼存ê浜鹃柛锔诲幖娴犳帞绱掗煬鎻掑姦闁绘侗鍠氶幑鍕传閸曨厺娣┑鐘愁問閸犳牠顢栭崨顖楀亾?    ALOGN("[AIProcessor] Creating model instance...");
    IAIModel* newModel = create();
    if (!newModel) {
        ALOGE("Failed to create model instance");
        dlclose(handle);
        return false;
    }
    
    ALOGN("[AIProcessor] Initializing model: %s", modelPath.c_str());
    int initRet = newModel->Init(modelPath.c_str());
    if (initRet != 0) {
        ALOGE("Model Init failed: %s, ret=%d", modelPath.c_str(), initRet);
        // 闂備礁鎲＄敮妤冩崲閸岀儑缍栭柟鐗堟緲缁€宀勬煛瀹擃喕妞掗弶顓㈡煟閻樺啿濮傞柛搴㈠絻椤啴宕掗悙瀵稿弳闂侀€炲苯澧悮娆徝归敐澶屽濞撴埃鍋撶€规洖婀遍埀顒婄秵娴滅偤寮堕挊澶堚偓?
        DestroyAIModelFunc destroy = (DestroyAIModelFunc)dlsym(handle, "DestroyAIModel");
        if (destroy) destroy(newModel);
        dlclose(handle);
        return false;
    }
    
    ALOGN("[AIProcessor] Model initialized successfully");

    // 濠电儑绲藉ú锔炬崲閸岀偞鍋ら柕濞垮剻閻旂厧鐏崇€规洖娲犻崑鎾搭槹鎼存ê浜鹃柛锔诲幖娴犳帞绱掗煬鎻掑姦鐎规洦浜炴禒锕傚箚瑜忕粊閿嬬箾鐎电鞋婵炲绋栭妵鎰版倷閸濆嫯鎽?
    model_ = newModel;
    pluginHandle_ = handle;
    modelPath_ = modelPath;
    if (!modelName.empty()) modelName_ = modelName;

    // 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉仜閻旂厧鐏崇€规洖娲犻崑鎾绘煥鐎ｎ偆绉堕梺瑙勫劤閸熷灝顕ｉ幎鑺ュ€堕煫鍥ㄦ尰閹牊銇勯弴銊ュ籍闁轰礁绉瑰畷濂告偄閸涘﹥鐣奸梻浣筋潐濠㈡ɑ顨ヨ箛鏇燁潟闁哄顑欏浼存煏閸繃顥炴い搴☆槺閳ь剝顫夐悺鏇犱焊濞嗘垹鐭欏鑸靛姇缁犳澘霉閿濆牜娼愮紒?modelMutex_ 闂傚倷娴囬～澶愭偋椤撶姵顫曟繝闈涱儏缁犮儵鏌嶈閸撴氨绮嬮幒妤€唯闁靛牆娲ㄩ幉褰掓⒑閻撳骸鏆遍柍褜鍓涢崳銉╁焵椤掑倸浠遍柟?getInputSize闂?
    // getInputSize 闂備胶顭堥弲顖炲炊瑜嶆慨銈嗙箾閹寸偞鈷掗柛鐘冲姇閵嗘帗绻濆顒€绨ュ┑掳鍊曠€氀囧绩閵堝鐓熸繛鎴炵懃缁茶崵鎲搁弶鎸庡枠闁哄瞼鍠栧浠嬫偨閻㈡妲烽梻浣告惈閻楀棝宕锔藉剭妞ゆ劧闄勯崵濠囨倵濞戞瑱渚涢柛鏂诲劦濮?
    // 闂備胶鍎甸弲娑㈡偤閵娧勬殰闁哄鍩堥崵?model_ 闂備胶绮粙鎺曘亹閸愵厹浜归柛銉ｅ妽缂嶆挾绱掔€ｎ亞浠㈢€殿喗濞婇幃妯跨疀閹惧瓨鍎撳?    int w = 640, h = 640;
    if (model_ && newModel) {
        model_->GetInputSize(&w, &h);
    }
    ALOGN("AI Model Loaded: %s, %dx%d", modelPath.c_str(), w, h);

    return true;
}

void AIProcessor::applyModelParamsToEnv(const std::string&, const nlohmann::json& params) {
    if (!params.is_object()) return;
    auto setFloat = [&](const char* key, const char* envKey) {
        if (!params.contains(key)) return;
        try {
            const std::string value = std::to_string(params[key].get<float>());
            setenv(envKey, value.c_str(), 1);
        } catch (...) {
            ALOGW("[AIProcessor] Ignore invalid parameter: %s", key);
        }
    };
    setFloat("conf_threshold", "MANHOLE_CONF_THRESH");
    setFloat("nms_threshold", "MANHOLE_NMS_THRESH");
}

void AIProcessor::unloadModel() {
    std::lock_guard<std::mutex> lock(modelMutex_);
    
    if (model_) {
        // 闂備胶顭堢换鎰版偋韫囨稒鍋ら柟瀛樼箥閸ゆ鏌涘☉鍗炲妞ゅ繘浜堕幃妯跨疀閹惧墎顔夊銈嗘煥閻倸顕ｉ鈧畷濂稿即閻曞倻鍚归梻浣瑰缁嬫垿鎯夋總绋跨伋婵☆垵宕甸埞宥嗙節闂堟稒宸濇慨濠囩畺閺岋繝宕煎┑鎰у銈嗗笚缁挸鐣烽悜钘壩╅柨鏇楀亾闁抽攱妫冮幃璺衡槈濡偐浼囧┑鐐茬墛閸ㄥ潡鐛幒妤€惟鐟滃酣宕?        usleep(50 * 1000);  // 50ms
        
        model_->Deinit(); // 婵犵妲呴崹顏堝焵椤掆偓绾绢參鍩€椤掑倸鍘寸€规洩绲介濂稿川椤撶姳娣┑鐘愁問閸犳牠顢栭崨顖楀亾?
        if (pluginHandle_) {
            // 闂佽崵濮撮鍛村疮娴兼潙鏋侀柕鍫濐槸缁犵敻鏌熼崫鍕ｆい鎰閳藉骞欓崘銊ョ濠电偛鐗婇崹鍧楀蓟閵娾晛鐒垫い鎺嗗亾鐞氭瑥霉閿濆浂鐒炬慨锝嗗姍閺屸剝鎷呭畡鏉跨ギ闂侀潻绲鹃幐鍐差嚕閵娾晜鍊锋い鎴犲枍缁舵艾鐣烽崷顓涘亾閿濆簼绨介柡澶庢閵?
            DestroyAIModelFunc destroy = (DestroyAIModelFunc)dlsym(pluginHandle_, "DestroyAIModel");
            if (destroy) destroy(model_);
        }

        model_ = nullptr;
    }

    if (pluginHandle_) {
        dlclose(pluginHandle_); // 闂備礁鎲￠〃鍡涙偤閺囩伝褰掑炊椤掆偓缁€澶愭煏婵犲繗鍚傞柛瀣尰閹峰懘宕妷锔炬
        pluginHandle_ = nullptr;
    }

    ALOGN("AI Model Unloaded");
}

// 闂佽娴烽弫鎼併€佹繝鍕偨妞ゆ挶鍨圭粈鍌炴煏婢跺牆鍔滃┑顔奸叄瀵爼鍩￠崒婊庣伇濡炪們鍨洪崹鍧楃嵁閹烘惟鐟滃酣宕愰悙宸唵閻犲搫鎼顐︽煙?bool AIProcessor::processFrame(const AX_VIDEO_FRAME_T* frame, AI_RESULT_T* result) {
    if (!frame || !result) return false;
    
    // 缂傚倷鐒﹀畷妯衡枖閺囥垹鐓濋柤娴嬫杹閸嬫捇鎮烽柇锔叫﹂梺鍛婄懃缁绘ê鐣烽悜钘夌閻忕偟鏅鍡涙⒑閸涘﹦鎳勯柣妤侇殙閸燁垶姊洪幖鐐插妞ゆ垵妫涢埀顒€鐏氬畝鎼佸蓟?
    IAIModel* currentModel = nullptr;
    {
        std::lock_guard<std::mutex> lock(modelMutex_);
        currentModel = model_;
        if (!currentModel) {
            return false;
        }
    }
    
    // 婵犵數鍋涢ˇ顓㈠礉瀹ュ绀堝ù鐓庣摠閺咁剚鎱ㄥΟ铏癸紞缂佺姴缍婂娲箵閹烘埈娈紓浣诡殔椤︾敻鐛幇顓熷閻熸瑥瀚悾鎶芥⒒娴ｆ悶浠掔紒韬插€楀Σ鎰攽閸喎顎涢梺瀹犳〃缁€浣虹矈婵犳艾绾ч柣鎰ゴ閸嬫捇鎮㈤搹鍦毇缂傚倷绀侀鍛搭敄閸涜埇浜归柛娆忣槺椤╃兘鏌曟径鍫濆缂佺姴顭烽幃?
    // 濠电偠鎻徊钘壩涘▎鎾冲瀭婵犲﹤瀚々閿嬨亜閹哄棗浜鹃梺鍛婂煀缁辨洜妲愰幒鏇犵杸閹艰揪绲块鏃堟煟鎼淬垻鈯曢惇澶岀磼閵娾懇鍋撳鍕枛閸ㄦ儳鐣烽崶锝呬壕闁绘垼妫勭粻浼存煕閵夘喖澧悗姘洴閺屻劌鈽夊Ο鍨伃閻熸粍濡搁崨顔兼毇婵炶揪绲介幉锟犵叕椤掑嫭鐓欓柟顓熷笒婵″潡鏌熼纭峰姛闁逞屽墰閹虫捇寮甸鍕瀭婵娉涚粈鍫⑩偓骞垮劚鐎氼噣鎮峰┑瀣拺闁圭粯甯弨濠氭煛閸屾瑨鍏屽ù婊勬倐瀹曪繝鎮欓懠棰濆敼濠?
    
    // 濠电姰鍨煎▔娑氣偓姘煎櫍楠炲啯绻濋崶褏顦ㄩ梺鍛婁緱閸橀箖骞楅悩缁樼厸闁告洦鍘煎瓭闂?    AX_BOOL bMapped = AX_FALSE;
    AX_VIDEO_FRAME_T tFrame = *frame;
    
    if (!tFrame.u64VirAddr[0] && tFrame.u64PhyAddr[0]) {
        tFrame.u64VirAddr[0] = (AX_U64)AX_SYS_Mmap((AX_U64)tFrame.u64PhyAddr[0], tFrame.u32FrameSize);
        bMapped = AX_TRUE;
    }

    int ret = currentModel->Inference(&tFrame, result);
    
    if (bMapped && tFrame.u64VirAddr[0]) {
        AX_SYS_Munmap((void*)tFrame.u64VirAddr[0], tFrame.u32FrameSize);
    }
    
    return (ret == 0);
}

// 闂佽崵濮崇粈浣规櫠娴犲鍋柛鈩冪懅绾剧偓銇勯弮鍥跺敽缂佽妫濋獮鏍ㄦ綇閸撗呮殸闂佸綊顥撳▍涓甋闂傚倸鍊搁崯顐︽偋閸℃稑鐒?
void AIProcessor::setThresholds(float conf, float nms) {
    std::lock_guard<std::mutex> lock(modelMutex_);
    confThreshold_ = conf;
    nmsThreshold_ = nms;
    ALOGD("AI thresholds updated: conf=%.2f, nms=%.2f", conf, nms);
}

// 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉仜閻旂厧鐏崇€规洖娲犻崑鎾绘煥鐎ｎ偆绉堕梺瑙勫劤閸熷灝顕ｉ幎鑺ュ€堕煫鍥ㄦ尰閹牊銇?void AIProcessor::getInputSize(int* w, int* h) const {
    // [Optim] 缂傚倷绀侀ˇ顖炩€﹀畡鎵虫瀺閹兼番鍔岀粈鍕煠閹帒鍔滄繛鍫濈埣閺屾稖绠涚€ｎ亜濮庨悷?闂傚倷绀侀妵妯好归崶顒傚祦闊洦绋戦惌妤呮煛瀹ュ骸浜為柟鑲╁帶铻栭柛灞惧喕閼版寧銇?Debug 闂備礁鎼崯銊╁磿鏉堚晜宕查柡鍐ㄧ墛閺咁剛鈧厜鍋撻柛鎰典簽椤旀劖绻涚€涙鐭嬬紒瀣笒铻為柍鍝勫閻も偓濠德板€愰崑鎾寸節閳ь剟顢旈崟鎴炲笩缁犳盯骞橀弶鎴經婵犵數鍋為幐绋款嚕閸洘鍋傞悗锝庡枛缁秹鏌￠崼銏℃毄缂佹劖顨婇弻娑㈠箳閹存繃些闂佹椿鍋勫Λ婊堝Φ閹版澘纭€闁绘劕鐡ㄩ崕銉╂⒑鐠団€冲姱闁糕剝顨呴獮?    std::lock_guard<std::mutex> lock(modelMutex_);
    if (model_ && w && h) {
        model_->GetInputSize(w, h);
    } else {
        // 濠电偛顕慨鎾箠鎼粹槄鑰挎い蹇撳閸ゆ洟鏌涚仦鐐殤闁糕晜顨婇弻鐔碱敍濮橆厼娅ｉ梺鍝ュТ閻倿鐛澶婄闁告劘灏欓妶顏堟煟閻橆偄浜鹃梺鎸庢磵閸嬫捇鏌熺拠褏绡€闁轰礁绉舵禒锕傛嚃閳哄啯鍣搁梻鍌氬€哥€氼參宕濋弴銏犳槬婵°倓绶ょ槐锝夋煙鐎电孝闁肩儤濞婇弻銊モ槈濡厧鈪遍梺浼欑稻缁诲牆鐣烽敓鐘插嵆闁绘ê纾弳鐘充繆椤愩倕顣奸柛銊ф嚀椤斿繑绺界粙璺唺閻熸粌绉瑰畷鍝勎旈崨顔芥珫?
        static int warn_count = 0;
        if (warn_count++ % 1000 == 0) {
             ALOGW("[AIProcessor] Model not available when getting input size, defaulting to 640x640");
        }
        if (w) *w = 640;
        if (h) *h = 640;
    }
}

// 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉簵娴滃綊鏌熼幆褍鏆辨い銈呮噺娣囧﹪顢涘璇蹭壕鐎规洖娲犻崑鎾搭槹鎼淬垹纾銈嗙墬濮樸劎绮?std::string AIProcessor::getModelPath() const { 
    std::lock_guard<std::mutex> lock(modelMutex_);
    return modelPath_; 
}

// 闂備胶绮粙鎺曘亹閸愵厹浜?OSD 婵犵數鍋為幐绋款嚕閸洘鍋傞悗锝庡枛闂傤垱銇勯鐔风缂佲偓閸曨剚鍙忛柨婵嗘噽缁犱即鏌嶈閸撴瑩顢栭崱姘殲闂備焦鎮堕崝宀勵敄閸涜埇浜归柛娆忣槺椤╃兘鏌曟径娑㈡缂佺姵甯￠弻銈嗙附婢跺鐩庣紓浣筋唺缁舵岸骞嗛崘顔肩妞ゆ帒鍠氬ú顒勬⒑閸濆嫭婀伴柟鍝ヮ焾閳诲秹骞掑Δ浣规珫?
std::shared_ptr<IOSDRenderer> AIProcessor::getOSDRenderer() {
    std::lock_guard<std::mutex> lock(modelMutex_);
    if (!osdRenderer_) {
        osdRenderer_ = std::make_shared<DefaultOSDRenderer>();
    }
    return osdRenderer_;
}










