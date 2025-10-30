import math
from pathlib import Path

from dm_control import mjcf
from dm_control import composer

from cathsim.dm.observables import JointObservables
from cathsim.dm.components.base_models import BaseGuidewire
from cathsim.dm.utils import get_env_config

# 把 env.yaml 里的 guidewire: 段 和 guidewire.yaml 全文 深度合并，因此你能同时在两个文件里写参数，后者优先级更高
guidewire_config = get_env_config("guidewire")
guidewire_default = guidewire_config["default"]

tip_config = get_env_config("tip")
tip_default = tip_config["default"]

SCALE = guidewire_config["scale"]
RGBA = guidewire_config["rgba"]
BODY_DIAMETER = guidewire_config["diameter"]


# def get_body_properties(
#     scale: float, body_diameter: float, sphere_to_cylinder: float = 1.5
# ):
def get_body_properties(
            scale: float, body_diameter: float, sphere_to_cylinder: float = 1.5
    ):
    """计算导丝体的物理属性
        参数:
            scale: 缩放比例
            body_diameter: 主体直径
            sphere_to_cylinder: 球体与圆柱体的比例系数（默认1.5）
        返回:
            球体半径、圆柱体高度、偏移量的元组
        """
    sphere_radius = (body_diameter / 2) * scale # 球体半径
    cylinder_height = sphere_radius * sphere_to_cylinder # 圆柱体高度
    offset = sphere_radius + cylinder_height * 2 # 偏移量（体之间的距离）
    return sphere_radius, cylinder_height, offset


SPHERE_RADIUS, CYLINDER_HEIGHT, OFFSET = get_body_properties(
    scale=guidewire_config["scale"],
    body_diameter=guidewire_config["diameter"],
)


def add_body(n, parent, stiffness=None, name=None, OFFSET=0.1):
    """Add a body to the guidewire.
    Args:
        n (int): The index of the body to add
        parent (mjcf.Element): The parent element to add the body to
        stiffness (float): Stiffness of the joint
        name (str): Name of the body/joint/geom
    向导丝添加体
    参数:
        n: 要添加的体的索引
        parent: 父元素（用于挂载新体）
        stiffness: 关节刚度
        name: 体/关节/几何（体）的名称
        OFFSET: 偏移量（默认0.1）
    返回:
        添加的子体元素
    """
    child = parent.add("body", name=f"{name}_body_{n}", pos=[0, 0, OFFSET])    # 创建子体，位置基于偏移量
    child.add("geom", name=f"geom_{n}")# 添加几何体
    j0 = child.add("joint", name=f"{name}_J0_{n}", axis=[1, 0, 0])
    j1 = child.add("joint", name=f"{name}_J1_{n}", axis=[0, 1, 0])
    # j2 = child.add("joint", name=f"{name}_J2_{n}", axis=[0, 0, 1])


    # j0 = child.add("joint", name=f"{name}_J0_{n}")  # 添加两个关节（分别绕X轴和Y轴旋转）
    # j1 = child.add("joint", name=f"{name}_J1_{n}")  # 添加两个关节（分别绕X轴和Y轴旋转）

    # child.add("joint",
    #           type="universal",
    #           name=f"{name}_U{n}") \
    #     .add("axis", [1, 0, 0]) \
    #     .add("axis2", [0, 1, 0]) # hinge 是单轴合页，universal 是万向节

    if stiffness is not None:    # 若指定刚度，则设置关节刚度
        j0.stiffness = stiffness
        j1.stiffness = stiffness

    return child


class Guidewire(BaseGuidewire):
    """Guidewire class"""

    def _build(self, n_bodies: int = 80):
        """Build the guidewire.

        Set the default values, add bodies and joints, and add actuators.

        Args:
            n_bodies (int): Number of bodies to add to the guidewire
        构建导丝
        设置默认值，添加体和关节，以及添加执行器。
        参数:
            n_bodies: 要添加到导丝的体的数量
        """
        # print("n_bodies==",n_bodies)
        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies # 计算导丝总长度
        # print("self._length==",self._length)

        self._n_bodies = n_bodies # 体的数量

        self._mjcf_root = mjcf.RootElement(model="guidewire")  # 创建MJCF根元素（用于描述物理模型）
        # 初始化默认属性、体与关节、执行器
        self._set_defaults()
        self._set_bodies_and_joints()
        self._set_actuators()

    @property
    def mjcf_model(self):
        """返回导丝的MJCF模型"""
        return self._mjcf_root

    def _set_defaults(self):
        """Set the default values for the guidewire. 设置导丝的默认"""
        # 为所有geom设置默认属性（从YAML读取）
        self._mjcf_root.default.geom.set_attributes(
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],  # 计算得到的尺寸
            **guidewire_default["geom"]  # 展开YAML中的geom参数（如group、friction等）
        )

        # 为所有joint设置默认属性
        self._mjcf_root.default.joint.set_attributes(
            pos=[0, 0, -OFFSET / 2],  # 位置偏移
            **guidewire_default["joint"]  # 展开YAML中的joint参数（如type、stiffness等）
        )
        # 为所有位点（site）设置默认属性
        self._mjcf_root.default.site.set_attributes(
            size=[SPHERE_RADIUS],
            **guidewire_default["site"],
        )
        # 为所有速度执行器设置默认属性
        self._mjcf_root.default.velocity.set_attributes(
            **guidewire_default["velocity"],
        )

    def _set_bodies_and_joints(self):
        """Set the bodies and joints of the guidewire.设置导丝的体和关节"""
        # 添加根体（导丝的第一个体）
        parent = self._mjcf_root.worldbody.add(
            "body",
            name="guidewire_body_0",
            euler=[-math.pi / 2, 0, math.pi],# 修改导丝的初始朝向
            pos=[0, -(self._length - 0.015), 0],# 初始位置
        )
        # 为根体添加几何（体）
        parent.add(
            "geom",
            name="guidewire_geom_0",
        )
        # 添加滑动关节（控制导丝推进/后退）
        parent.add(
            "joint",
            type="slide",
            name="slider",
            range=[-0, 0.2],  # 滑动范围：0 到 0.2 米（即最大前进距离为0.2米）
            stiffness=0,  # 刚度（0表示无弹性约束）
            damping=2,  # 阻尼（阻碍运动的阻力）
        )
        # 添加旋转关节（控制导丝旋转）
        parent.add(
            "joint",
            type="hinge",
            name="rotator",
            stiffness=0,
            damping=2,
        )

        stiffness = self._mjcf_root.default.joint.stiffness  # 从默认关节刚度开始，逐段添加体和关节
        for n in range(1, self._n_bodies):# 一共添加_n_bodies个关节
            # 每添加一个体，刚度乘以0.995（使远离驱动端的关节更柔软）
            parent = add_body(n, parent, stiffness=stiffness, name="guidewire", OFFSET=OFFSET)
            stiffness *= 0.995 #导丝主体的关节刚度在逐段递减，使得远离驱动端的关节更柔软，整体呈现柔性
            # stiffness *= 1 #导丝主体的关节刚度在逐段递减，使得远离驱动端的关节更柔软，整体呈现柔性

        # 末端添加导丝尖端
        self._tip_site = parent.add("site", name="tip_site", pos=[0, 0, OFFSET])

    def _set_actuators(self):
        """设置导丝的执行器"""
        # kp = 40  # 比例增益，力矩：1 rad 角度误差 → 40 N·m 力矩
        kp = 40  # 比例增益，力矩
        # 添加推进、后退关节执行器
        self._mjcf_root.actuator.add(
            "velocity",
            joint="slider",
            name="slider_actuator",
        )
        # 添加左旋、右旋关节执行器
        self._mjcf_root.actuator.add(
            "general",
            joint="rotator",
            name="rotator_actuator",
            dyntype=None,  # 动态类型（无）
            gaintype="fixed",  # 增益类型（固定）
            biastype="None",  # 偏置类型（无）
            dynprm=[1, 0, 0],  # 动态参数
            gainprm=[kp, 0, 0],  # 增益参数
            biasprm=[0, kp, 0],  # 偏置参数
        )

    @property
    def attachment_site(self):
        """The attachment site of the guidewire. Useful for attaching the tip to the guidewire."""
        """导丝的附着位点，用于将尖端附着到导丝上。"""
        return self._tip_site

    def _build_observables(self):
        """Build the observables of the guidewire."""
        """构建导丝的观测器"""
        return JointObservables(self)

    def save_model(self, path: Path):
        """Save the guidewire model to an `.xml` file.
        Usefull for debugging, exporting, and visualizing the guidewire.
        Args:
            path (Path): Path to save the model to
        将导丝模型保存为.xml文件，用于调试、导出和可视化导丝。
        参数:
            path: 保存路径
        """
        if path.suffix is None:
            path = path / "guidewire.xml"
        with open(path, "w") as file:
            file.write(self.mjcf_model.to_xml_string("guidewire"))


class Tip(BaseGuidewire):
    def _build(self, name: str = "tip", n_bodies: int = 3):
        """Build the tip of the guidewire.
        Args:
            name (str): Name of the tip (Will be removed in the future)
            n_bodies (int): Number of bodies to add to the tip
        构建导丝的尖端
        参数:
            name: 尖端名称（未来将移除）
            n_bodies: 要添加到尖端的体的数量
        """
        self._mjcf_root = mjcf.RootElement(model=name)
        self._n_bodies = n_bodies

        self._setup_defaults()
        self._setup_bodies_and_joints()

        self.head_geom.name = "head"   # 将最后一个几何（体）命名为"head"（尖端头部）

    @property
    def mjcf_model(self):
        return self._mjcf_root # 返回尖端的MJCF模型

    def _setup_defaults(self):
        """Set the default values for the tip."""
        self._mjcf_root.default.geom.set_attributes(
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            **tip_default["geom"], # 从尖端的YAML配置读取geom参数
        )

        self._mjcf_root.default.joint.set_attributes(
            pos=[0, 0, -OFFSET / 2],  # 位置偏移
            springref=math.pi / 3 / self._n_bodies,  # 弹簧参考位置（初始角度）
            **tip_default["joint"],  # 从尖端的YAML配置读取joint参数
        )

    def _setup_bodies_and_joints(self):
        """Setup the bodies and joints of the tip.设置尖端的体和关节"""
        parent = self._mjcf_root.worldbody.add(
            "body",
            name="tip_body_0",
            euler=[0, 0, 0],  # 初始朝向
            pos=[0, 0, 0],  # 初始位置
        )
        # 添加几何（体）
        parent.add("geom", name="tip_geom_0")
        # 添加两个关节（分别绕Z轴和Y轴旋转）
        parent.add("joint", name="tip_J0_0", axis=[0, 0, 1])
        parent.add("joint", name="tip_J1_0", axis=[0, 1, 0])

        # 逐段添加尖端的体
        for n in range(1, self._n_bodies):
            parent = add_body(n, parent, name="tip", OFFSET=OFFSET)

    def _build_observables(self):
        """Setup the observables of the tip.设置尖端的观测器"""
        return JointObservables(self)

    @property
    def head_geom(self):
        """Get the head geom of the tip.获取尖端的头部几何（体）"""
        return self._mjcf_root.find_all("geom")[-1] # 返回最后一个几何（体）


if __name__ == "__main__":
    guidewire = Guidewire()
    guidewire.mjcf_model.compiler.set_attributes(autolimits=True, angle="radian")
    mjcf.Physics.from_mjcf_model(guidewire.mjcf_model)
    guidewire.save_model(Path.cwd() / "guidewire.xml")
