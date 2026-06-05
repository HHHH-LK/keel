"""app 层接线:ChatCLI 持有会话级 PermissionGrants。"""
from my_agent_llms.cli.app import ChatCLI, load_config
from my_agent_llms.cli.permission_grants import PermissionGrants


def test_chatcli_has_permission_grants():
    cli = ChatCLI(load_config())
    assert isinstance(cli.grants, PermissionGrants)
