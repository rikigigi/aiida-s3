"""Generic export/import of Data nodes and Group subclasses as shared S3 collections.

This module provides a simple, provenance-free way to share Data nodes (with their
repository files) and Group subclasses between AiiDA profiles. The shared objects
are referenced by key in a read-only S3 bucket - no data is duplicated.

Manifest format:
    {
        "format": "aiida-s3-shared-collection/1",
        "source": {"endpoint_url": ..., "bucket_name": ...},
        "data_nodes": [
            {
                "source_uuid": ...,
                "node_type": "pseudo.upf",  # entry point name
                "attributes": {...},
                "extras": {...},
                "label": ...,
                "repository_files": [
                    {"path": "Si.UPF", "s3_key": "uuid-...", "size_bytes": ..., "sha256": ...}
                ]
            }
        ],
        "groups": [
            {
                "source_uuid": ...,
                "group_type": "pseudo.family.sssp",  # entry point name
                "label": "SSSP/1.3/PBE/efficiency",
                "description": ...,
                "extras": {...},  # includes cutoffs, etc.
                "member_source_uuids": ["uuid1", "uuid2"]  # references to data_nodes
            }
        ]
    }

Usage:
    # Export a pseudo family
    export_shared_collection(
        data_nodes=[pseudo1, pseudo2, ...],
        groups=[family],
        output=Path("family.json"),
        shared_name="common"
    )

    # Import into another profile
    import_shared_collection(
        manifest_path=Path("family.json"),
        shared_name="common"
    )
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import typing as t

__all__ = (
    'COLLECTION_FORMAT',
    'export_shared_collection',
    'import_shared_collection',
    'iter_repository_keys',
)

COLLECTION_FORMAT = 'aiida-s3-shared-collection/1'


def iter_repository_keys(metadata: dict[str, t.Any], prefix: str = '') -> t.Iterator[tuple[str, str]]:
    """Yield ``(path, key)`` tuples for every file entry in node repository metadata."""
    if 'k' in metadata:
        yield prefix, metadata['k']
    for name, child in metadata.get('o', {}).items():
        yield from iter_repository_keys(child, f'{prefix}/{name}' if prefix else name)


def _get_entry_point_name(obj: t.Any) -> str:
    """Return the entry point name for a Data or Group class."""
    if hasattr(obj, 'get_entry_point_name'):
        return obj.get_entry_point_name()
    # Fallback: get from class
    cls = obj if isinstance(obj, type) else obj.__class__
    from aiida.plugins.entry_point import get_entry_point_from_class
    _, entry_point = get_entry_point_from_class(cls.__module__, cls.__name__)
    return entry_point.name


def export_shared_collection(
    data_nodes: t.Iterable[t.Any],
    groups: t.Iterable[t.Any] | None = None,
    output: pathlib.Path | str = 'shared_collection.json',
    shared_name: str = 'shared',
    verify: bool = True,
) -> pathlib.Path:
    """Export Data nodes and Groups as a shared collection manifest.

    The manifest contains all metadata needed to recreate the nodes in another
    profile, with repository files referenced by key in the source S3 bucket.

    :param data_nodes: iterable of Data nodes to export.
    :param groups: iterable of Group nodes to export (with their members).
    :param output: path to write the manifest JSON file.
    :param shared_name: name under which the source bucket will be registered
        as a shared repository in the importing profile.
    :param verify: whether to verify that all repository objects exist in S3.
    :return: the path to the created manifest file.
    """
    from aiida import orm
    from aiida.manage import get_manager

    output = pathlib.Path(output)
    backend = get_manager().get_profile_storage()
    repository = backend.get_repository()
    profile = backend.profile

    data_nodes = list(data_nodes)
    groups = list(groups or [])

    # Collect all S3 keys for verification
    all_keys: list[str] = []

    manifest: dict[str, t.Any] = {
        'format': COLLECTION_FORMAT,
        'shared_name': shared_name,
        'source': {
            'endpoint_url': profile.storage_config.get('endpoint_url'),
            'bucket_name': profile.storage_config['bucket_name'],
        },
        'data_nodes': [],
        'groups': [],
    }

    # Export data nodes
    for node in data_nodes:
        if not isinstance(node, orm.Data):
            raise TypeError(f'Expected Data node, got {type(node)}')
        if not node.is_stored:
            raise ValueError(f'Node {node} is not stored')

        # Get repository files
        repository_files = []
        metadata = node.backend_entity.repository_metadata or {}
        for path, key in iter_repository_keys(metadata):
            all_keys.append(key)
            file_info: dict[str, t.Any] = {
                'path': path,
                's3_key': key,
            }
            # Get file size and hash
            try:
                with repository.open(key) as handle:
                    content = handle.read()
                file_info['size_bytes'] = len(content)
                file_info['sha256'] = hashlib.sha256(content).hexdigest()
            except Exception:
                pass  # Size/hash optional
            repository_files.append(file_info)

        manifest['data_nodes'].append({
            'source_uuid': node.uuid,
            'node_type': _get_entry_point_name(node),
            'attributes': dict(node.base.attributes.all),
            'extras': dict(node.base.extras.all),
            'label': node.label or '',
            'description': node.description or '',
            'repository_files': repository_files,
        })

    # Verify all objects exist
    if verify and all_keys:
        missing = [key for key, exists in zip(all_keys, repository.has_objects(all_keys)) if not exists]
        if missing:
            raise FileNotFoundError(f'Objects missing from S3 bucket: {missing}')

    # Export groups
    for group in groups:
        if not isinstance(group, orm.Group):
            raise TypeError(f'Expected Group, got {type(group)}')
        if not group.is_stored:
            raise ValueError(f'Group {group} is not stored')

        # Get member UUIDs (only Data nodes that are in our export set)
        exported_uuids = {node.uuid for node in data_nodes}
        member_source_uuids = [
            node.uuid for node in group.nodes
            if node.uuid in exported_uuids
        ]

        manifest['groups'].append({
            'source_uuid': group.uuid,
            'group_type': _get_entry_point_name(group),
            'label': group.label,
            'description': group.description or '',
            'extras': dict(group.base.extras.all),
            'member_source_uuids': member_source_uuids,
        })

    output.write_text(json.dumps(manifest, indent=2))
    return output


def import_shared_collection(
    manifest_path: pathlib.Path | str,
    shared_name: str | None = None,
    credentials: dict[str, str] | None = None,
    label_prefix: str = '',
) -> dict[str, t.Any]:
    """Import a shared collection manifest into the current profile.

    Creates new Data nodes and Groups, with repository files referenced by key
    in the shared S3 bucket. No data is uploaded or downloaded.

    :param manifest_path: path to the manifest JSON file.
    :param shared_name: name of the shared repository. If None, uses the name
        from the manifest.
    :param credentials: optional credentials to register the shared repository
        if not already registered. Dict with ``access_key_id`` and
        ``secret_access_key``.
    :param label_prefix: optional prefix to add to group labels (to avoid conflicts).
    :return: dict with ``data_nodes`` and ``groups`` lists of imported entities.
    """
    from aiida import orm
    from aiida.manage import get_manager
    from aiida.plugins import DataFactory, GroupFactory
    from aiida_s3.shared import add_shared_repository
    from aiida_s3.repository.multi import make_shared_key, MultiS3RepositoryBackend

    manifest_path = pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    if manifest.get('format') != COLLECTION_FORMAT:
        raise ValueError(f'Not a shared collection manifest: {manifest_path}')

    shared_name = shared_name or manifest.get('shared_name', 'shared')

    # Register shared repository if credentials provided
    if credentials:
        add_shared_repository(
            name=shared_name,
            endpoint_url=manifest['source']['endpoint_url'],
            access_key_id=credentials['access_key_id'],
            secret_access_key=credentials['secret_access_key'],
            bucket_name=manifest['source']['bucket_name'],
        )

    backend = get_manager().get_profile_storage()
    repository = backend.get_repository()

    # Ensure we have a multi-repository backend
    if not isinstance(repository, MultiS3RepositoryBackend):
        if backend.profile.storage_config.get('shared_repositories'):
            # Rebuild repository
            backend._repository = None
            repository = backend.get_repository()
        if not isinstance(repository, MultiS3RepositoryBackend):
            raise ValueError('Profile does not support shared repositories (use s3.psql_s3 storage)')

    # Verify shared repository is accessible
    if not repository.verify_shared().get(shared_name, False):
        raise ValueError(
            f"Shared repository '{shared_name}' not registered or not reachable. "
            "Register it with 'aiida-s3 shared add' or pass credentials."
        )

    # Map source UUID -> imported node
    uuid_map: dict[str, orm.Data] = {}
    imported_data_nodes: list[orm.Data] = []
    imported_groups: list[orm.Group] = []

    # Import data nodes
    for node_data in manifest['data_nodes']:
        source_uuid = node_data['source_uuid']
        node_type = node_data['node_type']

        # Load the node class
        try:
            node_class = DataFactory(node_type)
        except Exception as e:
            raise ValueError(f'Cannot load node type {node_type}: {e}')

        # Build shared repository metadata first
        metadata: dict[str, t.Any] = {'o': {}}
        shared_keys: list[str] = []
        for file_info in node_data.get('repository_files', []):
            path = file_info['path']
            s3_key = file_info['s3_key']
            shared_key = make_shared_key(shared_name, s3_key)

            # Verify object exists in shared bucket
            if not repository.has_objects([shared_key])[0]:
                raise FileNotFoundError(
                    f"Object '{s3_key}' not found in shared bucket '{shared_name}'"
                )

            shared_keys.append(shared_key)

            # Build nested metadata structure
            parts = [p for p in path.split('/') if p]
            current = metadata['o']
            for part in parts[:-1]:
                current = current.setdefault(part, {}).setdefault('o', {})
            if parts:
                current[parts[-1]] = {'k': shared_key}

        # Strategy: read content from shared bucket, create node with real content,
        # store it, then redirect to shared S3 and delete from primary bucket
        import io

        if node_data.get('repository_files'):
            # Use first file's path as the filename
            first_file = node_data['repository_files'][0]
            filename = pathlib.Path(first_file['path']).name

            if issubclass(node_class, orm.SinglefileData):
                # Read content from shared bucket for node creation
                # (needed for UpfData parsing, then we redirect to shared S3)
                shared_key = shared_keys[0]
                with repository.open(shared_key) as handle:
                    content = handle.read()

                # Create with real content (parsing works correctly)
                node = node_class(io.BytesIO(content), filename=filename)
            else:
                node = node_class()
        else:
            node = node_class()

        # Set attributes (skip filename for SinglefileData - set by constructor)
        for key, value in node_data.get('attributes', {}).items():
            if key != 'filename':
                node.base.attributes.set(key, value)

        # Set extras
        for key, value in node_data.get('extras', {}).items():
            node.base.extras.set(key, value)

        # Set label/description
        if node_data.get('label'):
            node.label = node_data['label']
        if node_data.get('description'):
            node.description = node_data['description']

        # Store the node (uploads placeholder to primary bucket)
        node.store()

        # Now redirect repository to shared S3 (no data duplication)
        if shared_keys:
            # Replace repository_metadata with shared keys
            node.backend_entity.repository_metadata = metadata

            # Reset the repository cache so it rebuilds from new metadata
            node.base.repository._repository_instance = None

            # Delete the placeholder objects from primary bucket
            primary_repo = repository._primary if isinstance(repository, MultiS3RepositoryBackend) else repository
            primary_keys = list(primary_repo.list_objects())
            keys_to_delete = [k for k in primary_keys if k not in shared_keys]
            if keys_to_delete:
                try:
                    primary_repo.delete_objects(keys_to_delete)
                except Exception:
                    pass  # Best effort cleanup

        uuid_map[source_uuid] = node
        imported_data_nodes.append(node)

    # Import groups
    for group_data in manifest['groups']:
        group_type = group_data['group_type']

        # Load the group class
        try:
            group_class = GroupFactory(group_type)
        except Exception as e:
            raise ValueError(f'Cannot load group type {group_type}: {e}')

        # Create new group
        label = label_prefix + group_data['label'] if label_prefix else group_data['label']
        group = group_class(label=label)
        if group_data.get('description'):
            group.description = group_data['description']

        # Set extras
        for key, value in group_data.get('extras', {}).items():
            group.base.extras.set(key, value)

        # Store the group
        group.store()

        # Add members (only those that were exported)
        members = [
            uuid_map[uuid] for uuid in group_data.get('member_source_uuids', [])
            if uuid in uuid_map
        ]
        if members:
            group.add_nodes(members)

        imported_groups.append(group)

    return {
        'data_nodes': imported_data_nodes,
        'groups': imported_groups,
        'uuid_map': uuid_map,
    }
